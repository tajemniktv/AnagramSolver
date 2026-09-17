"""Shared JSON-schema fixtures under Python's independent validator."""
import json
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/".codex/temp/schema-python"))
from jsonschema import Draft202012Validator

def main():
    fixtures=json.loads((ROOT/"contracts/fixtures.json").read_text(encoding="utf-8"))
    for fixture in fixtures:
        schema=json.loads((ROOT/"contracts/generated"/f"{fixture['schema']}.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert Draft202012Validator(schema).is_valid(fixture["value"])==fixture["valid"],fixture
    executable=ROOT/"target/debug"/("anagram-cli.exe" if sys.platform=="win32" else "anagram-cli")
    request=next(f["value"] for f in fixtures if f["schema"]=="GenerateRequest" and f["valid"])
    output=subprocess.run([str(executable),"generate",str(ROOT/"tests/parity/dictionary.txt")],input=json.dumps(request),text=True,capture_output=True,check=True)
    schema=json.loads((ROOT/"contracts/generated/Generated.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(json.loads(output.stdout))
    print(f"Python schemas passed {len(fixtures)} shared fixtures")
if __name__=="__main__":main()
