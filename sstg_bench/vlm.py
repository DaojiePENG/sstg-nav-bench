"""OpenAI-compatible VLM annotation and auditable node-label evaluation."""
import argparse, base64, json, os, re, time
from pathlib import Path
import requests

CATEGORIES = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]

def _json_from_text(raw):
    raw = raw.strip().replace("```json", "").replace("```", "")
    m=re.search(r"\{.*\}",raw,re.S)
    if not m: raise ValueError("no JSON object in model output: "+raw[:500])
    return json.loads(m.group(0))

def annotate(image: Path, api_key: str, base_url: str, model: str, retries=6):
    """Annotate one RGB/panorama through the Responses API.

    gpt-5.5 on the configured PeterAI gateway uses the Responses wire API.
    Chat Completions intermittently returned an empty assistant message, which is
    why this function deliberately uses /responses and records the full payload.
    """
    encoded=base64.b64encode(image.read_bytes()).decode()
    prompt=("You label a 2x2 indoor panorama for ObjectNav. The TOP-LEFT tile (yaw +0) is the node's anchor view; "
            "the other tiles are context. Allowed categories: "+", ".join(CATEGORIES)+". Return JSON only: "
            '{"primary_category":"category or null","primary_confidence":0.0,'
            '"detections":[{"category":"...","confidence":0.0}],"visible_categories":[...],"description":"..."}. '
            "primary_category must be the single clearest, most central ObjectNav object in the TOP-LEFT anchor, not merely "
            "an object elsewhere in the room. Use null when no allowed object is clear. detections may include context objects. "
            "Confidence is 0 to 1. Map television/monitor to tv_monitor.")
    body={"model":model,"input":[{"role":"user","content":[
        {"type":"input_text","text":prompt},{"type":"input_image","image_url":"data:image/jpeg;base64,"+encoded}]}],
        "reasoning":{"effort":"low"},"max_output_tokens":1000}
    last=None
    for attempt in range(retries):
        try:
            r=requests.post(base_url.rstrip('/')+"/responses",headers={"Authorization":"Bearer "+api_key,"Content-Type":"application/json"},json=body,timeout=180)
            r.raise_for_status(); payload=r.json(); chunks=[]
            for item in payload.get("output",[]):
                for content in item.get("content",[]):
                    if content.get("type")=="output_text": chunks.append(content.get("text",""))
            raw="\n".join(chunks)
            if not raw: raise ValueError("empty Responses output: "+json.dumps(payload)[:1000])
            parsed=_json_from_text(raw)
            parsed["_response_id"]=payload.get("id")
            parsed["_usage"]=payload.get("usage",{})
            return parsed,raw
        except Exception as e:
            last=e
            if attempt+1<retries: time.sleep(min(30,2**attempt))
    raise last

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--maps",default="outputs/hm3d_minival_oracle/maps")
    ap.add_argument("--output",default="outputs/hm3d_minival_oracle/vlm_audit.json")
    ap.add_argument("--limit",type=int,default=12); ap.add_argument("--model",default="gpt-5.5")
    ap.add_argument("--base-url",default="https://api.peterai.cc.cd/v1"); args=ap.parse_args()
    key=os.getenv("PeterAI_KEY");
    if not key: raise SystemExit("PeterAI_KEY is not set")
    images=sorted(Path(args.maps).glob("*/observations/*.jpg"))
    # Stratified first occurrence per category, then deterministic fill.
    chosen=[]
    for cat in CATEGORIES:
        p=next((x for x in images if x.stem.endswith('_'+cat)),None)
        if p: chosen.append(p)
    chosen=(chosen+[p for p in images if p not in chosen])[:args.limit]
    rows=[]
    for p in chosen:
        truth=p.stem.split('_',2)[-1]; started=time.time()
        try:
            parsed,raw=annotate(p,key,args.base_url,args.model); pred=parsed.get("visible_categories",[])
            rows.append({"image":str(p),"oracle_category":truth,"predicted":pred,"hit":truth in pred,
                         "description":parsed.get("description",""),"latency_s":round(time.time()-started,3),"raw":raw})
        except Exception as e: rows.append({"image":str(p),"oracle_category":truth,"predicted":[],"hit":False,"error":repr(e)})
    report={"model":args.model,"samples":len(rows),"recall_at_node":sum(r["hit"] for r in rows)/len(rows),"rows":rows}
    Path(args.output).write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=="__main__": main()
