// Execute only the actual checked-in component script against finite API data.
const fs = require('fs');
const [scriptPath, inputPath] = process.argv.slice(2);
const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const script = fs.readFileSync(scriptPath, 'utf8');
const gets = [], updates = [], pagination = [];
const source = 'a'.repeat(40), correlation = 'fixture-correlation';
Object.assign(process.env, {
  EXPECTED_LANE_CHECKS: JSON.stringify(input.expected ?? ['CI / Linux', 'CI / macOS']),
  SOURCE_SHA: input.source ?? source,
  CORRELATION_ID: input.correlation ?? correlation,
  LANE_NAME: 'CI / Linux', LANE_RESULT: input.result ?? 'success',
  LANE_CHECK_ID: '7', OVERALL_CHECK_ID: input.overall === false ? '' : '9', PLAN_DIGEST: 'd'.repeat(64)
});
const checks = {
  7: {name: 'CI / Linux', external_id: correlation, head_sha: source},
  9: {name: 'CI Required', external_id: correlation, head_sha: source}
};
if (input.mutation) checks[input.mutation.id][input.mutation.key] = input.mutation.value;
const github = {
  rest: {checks: {
    get: async request => { gets.push(request); return {data: checks[request.check_run_id]}; },
    update: async request => { updates.push(request); return {data: {}}; },
    listForRef: function listForRef() {}
  }},
  paginate: async (method, request) => {
    if (method !== github.rest.checks.listForRef) throw new Error('wrong API method');
    pagination.push(request);
    const rows = [
      {name:'CI / Linux',external_id:correlation,status:'completed',conclusion:input.result ?? 'success'},
      {name:'CI / macOS',external_id:correlation,status:input.incomplete?'in_progress':'completed',conclusion:input.peer_result ?? 'success'},
      {name:'CI / macOS',external_id:'other-correlation',status:'completed',conclusion:'failure'}
    ];
    return input.missing ? rows.filter(row=>row.name!=='CI / macOS'||row.external_id!==correlation):rows;
  }
};
const context = {repo:{owner:'fixture',repo:'local'},sha:'b'.repeat(40),runId:17,serverUrl:'https://example.invalid'};
const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
(async()=>{
  let error=null;
  try { await new AsyncFunction('github','context','setTimeout',script)(github,context,callback=>{callback();return 0;}); }
  catch(failure) { error=String(failure.message); }
  process.stdout.write(JSON.stringify({gets,updates,pagination,error,source,correlation,controller_sha:context.sha}));
})();
