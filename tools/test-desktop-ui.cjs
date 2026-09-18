// Exercise the actual component scripts with controlled IPC promises. No mocked
// solver or copied refresh implementation; these tests specifically cover races.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const ts = require('../apps/desktop/node_modules/typescript');
function deferred() { let resolve, reject; const promise = new Promise((a,b) => { resolve=a; reject=b; }); return {promise,resolve,reject}; }
const flush = () => new Promise(setImmediate);
function harness(component) {
  const first=deferred(), next=deferred(), operation=deferred(); let reads=0, mount;
  const source=fs.readFileSync(path.join(__dirname, '../apps/desktop/src', component+'.svelte'),'utf8').split('<script lang="ts">')[1].split('</script>')[0].replace(/^\s*import .*;$/gm,'');
  const state=Object.assign(x=>x,{snapshot:structuredClone});
  const bridge={maintenanceStatus:()=>++reads===1?first.promise:next.promise,train:()=>operation.promise,manage:()=>operation.promise};
  const context=vm.createContext({$state:state,$bindable:x=>x,$props:()=>({busy:false,corpora:{},settings:{},diagnostics:false}),bridge,message:String,setTimeout:()=>1,clearTimeout:()=>{},onMount:fn=>mount=fn});
  const expose=component==='Experimental' ? 'train,build,rank,get status(){return status},get buildReport(){return buildReport},get rankingReport(){return rankingReport}' : 'run,get report(){return report}';
  const compiled=ts.transpileModule(source+`\nglobalThis.api={${expose},get busy(){return busy},get error(){return error}};`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
  vm.runInContext(compiled,context); const dispose=mount();
  return {api:context.api,first,next,operation,dispose};
}
for(const action of ['train','build','rank']) {
  for(const rejected of [false,true]) test(`Experimental ${action} ignores stale ${rejected?'failure':'status'}`,async()=>{
    const h=harness('Experimental');
    const pending=h.api[action]({preventDefault(){}});
    assert.equal(h.api.busy,true);
    if(rejected) h.first.reject(new Error('previous request failed'));
    else h.first.resolve({active:false,error:'old error',report:{dataset_path:'old.json',indices:[1],model_path:'old-model.json'}});
    await flush();
    assert.equal(h.api.busy,true); assert.equal(h.api.error,''); assert.equal(h.api.status,null);
    assert.equal(h.api.buildReport,null); assert.equal(h.api.rankingReport,null);
    h.operation.resolve(); await flush();
    h.next.resolve({active:true,error:null,report:null}); await pending;
    assert.equal(h.api.busy,true); assert.equal(h.api.status.active,true); h.dispose();
  });
}
test('Management ignores a stale mount response while starting work',async()=>{
  const h=harness('Management'); const pending=h.api.run({kind:'prepare'});
  h.first.resolve({active:false,error:'old failure',report:{old:true}}); await flush();
  assert.equal(h.api.busy,true); assert.equal(h.api.error,''); assert.equal(h.api.report,null);
  h.operation.resolve(); await flush(); h.next.resolve({active:true,error:null,report:null}); await pending;
  assert.equal(h.api.busy,true); h.dispose();
});
test('Disposed Experimental refresh cannot publish an error',async()=>{
  const h=harness('Experimental'); h.dispose(); h.first.reject(new Error('late failure')); await flush();
  assert.equal(h.api.error,'');
});
