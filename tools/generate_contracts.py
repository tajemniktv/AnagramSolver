"""Generate schemas and structural TypeScript from Rust; --check rejects drift."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def typescript(schema):
    if schema is True:return "unknown"
    if schema is False:return "never"
    if "$ref" in schema:
        ref=schema["$ref"]
        if not ref.startswith("#/$defs/"):raise ValueError(f"Unsupported schema reference: {ref}")
        return ref.removeprefix("#/$defs/")
    if "const" in schema:return json.dumps(schema["const"])
    if "enum" in schema:return " | ".join(map(json.dumps,schema["enum"]))
    for keyword,operator in (("anyOf"," | "),("oneOf"," | "),("allOf"," & ")):
        if keyword in schema:return operator.join(f"({typescript(s)})" for s in schema[keyword])
    kind=schema.get("type")
    if isinstance(kind,list):return " | ".join(typescript(dict(schema,type=t)) for t in kind)
    if kind in ("string","boolean","null"):return kind
    if kind in ("integer","number"):return "number"
    if kind=="array":return f"Array<{typescript(schema['items'])}>"
    if kind=="object":
        fields=[f"  {json.dumps(k)}{'' if k in schema.get('required',[]) else '?'}: {typescript(v)};" for k,v in sorted(schema.get("properties",{}).items())]
        additional=schema.get("additionalProperties",True)
        patterns=schema.get("patternProperties",{})
        if patterns:
            if set(patterns)!={r"^\d+$"}:raise ValueError(f"Unsupported schema key patterns: {patterns}")
            # TypeScript cannot express an arbitrary key regex. Preserve the
            # value type; JSON Schema remains authoritative for numeric keys.
            fields.append(f"  [key: string]: {typescript(patterns[r'^\d+$'])};")
        if additional is not False:fields.append(f"  [key: string]: {typescript(additional)};")
        return "{\n"+"\n".join(fields)+"\n}"
    raise ValueError(f"Unsupported schema node: {schema}")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true");args=parser.parse_args()
    run=subprocess.run(["cargo","run","--quiet","--locked","--example","export_schema"],cwd=ROOT,text=True,capture_output=True,check=True)
    schemas=json.loads(run.stdout)
    outputs={}
    for name,schema in sorted(schemas.items()):
        outputs[ROOT/"contracts/generated"/f"{name}.schema.json"]=json.dumps(schema,indent=2,sort_keys=True)+"\n"
        lines=["// Generated from Rust by tools/generate_contracts.py. Do not edit."]
        for key,value in sorted(schema.get("$defs",{}).items()):lines.append(f"export type {key} = {typescript(value)};")
        lines.append(f"export type {name} = {typescript(schema)};")
        outputs[ROOT/"contracts/generated"/f"{name}.ts"]="\n\n".join(lines)+"\n"
    directory = ROOT / "contracts/generated"
    stale = set(directory.glob("*.schema.json")) | set(directory.glob("*.ts"))
    stale -= set(outputs)
    if stale:
        raise SystemExit("Stale generated artifacts; remove obsolete files explicitly: " +
                         ", ".join(str(path.relative_to(ROOT)) for path in sorted(stale)))
    for path,content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8")!=content:raise SystemExit(f"Generated contract drift: {path}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding="utf-8",newline="\n")
    print(f"{'Checked' if args.check else 'Generated'} {len(outputs)} schema/TypeScript artifacts")
if __name__=="__main__":main()
