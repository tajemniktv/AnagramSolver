import { readFileSync } from 'node:fs';
import Ajv from 'ajv/dist/2020.js';
const fixtures = JSON.parse(readFileSync(new URL('./fixtures.json', import.meta.url)));
const validators = new Map();
for (const fixture of fixtures) {
  if (!validators.has(fixture.schema)) {
    const schema = JSON.parse(readFileSync(new URL(`./generated/${fixture.schema}.schema.json`, import.meta.url)));
    // Rust integer/float formats are annotations; JSON numeric types are checked.
    validators.set(fixture.schema, new Ajv({ strict: false, validateFormats: false }).compile(schema));
  }
  const validate = validators.get(fixture.schema);
  if (validate(fixture.value) !== fixture.valid) throw new Error(JSON.stringify({ fixture, errors: validate.errors }));
}
console.log(`JavaScript schemas passed ${fixtures.length} shared fixtures`);
