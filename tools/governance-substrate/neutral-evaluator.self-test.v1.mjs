import assert from 'node:assert/strict';
import fs from 'node:fs';
import {evaluateNeutralEvidence,normalizeJiraScrumProfile} from './neutral-evaluator.v1.mjs';

const read=p=>JSON.parse(fs.readFileSync(new URL(p,import.meta.url),'utf8'));
const normal=read('../../control-plane/external-adapters/fixtures/JIRA_SCRUM_COHERENT_NORMAL_v1.json');
const mismatch=read('../../control-plane/external-adapters/fixtures/JIRA_SCRUM_DONE_APPROVED_IDENTITY_MISMATCH_v1.json');

const nr=evaluateNeutralEvidence(normalizeJiraScrumProfile(normal));
assert.equal(nr.result,'EVIDENCE_ELIGIBLE_FOR_CLOSURE_EVALUATION');
assert.equal(nr.primaryFailureClass,null);
assert.deepEqual(nr.consequentFailureClasses,[]);
assert.equal(nr.createsAuthority,false);

const mr=evaluateNeutralEvidence(normalizeJiraScrumProfile(mismatch));
assert.equal(mr.result,'FAIL_CLOSED');
assert.equal(mr.primaryFailureClass,'identity_conflict');
assert.deepEqual(mr.consequentFailureClasses,['continuity_unproven','closure_held']);
assert.equal(mr.terminalEffect,'CLOSURE_HELD_NO_TERMINAL_AUTHORITY');
assert.equal(mr.createsAuthority,false);

process.stdout.write(JSON.stringify({schema:'GOVERNANCE_SUBSTRATE_NEUTRAL_EVALUATOR_SELF_TEST_RECEIPT_v1',result:'PASS',controls:[
 {fixture:normal.fixtureId,result:nr.result,primaryFailureClass:nr.primaryFailureClass},
 {fixture:mismatch.fixtureId,result:mr.result,primaryFailureClass:mr.primaryFailureClass,consequences:mr.consequentFailureClasses}
]},null,2)+'\n');
