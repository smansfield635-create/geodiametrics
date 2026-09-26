import assert from 'node:assert/strict';
import fs from 'node:fs';
import {evaluateNeutralEvidence} from './neutral-evaluator.v1.mjs';
import {normalizeProcurementProfile} from './procurement-normalizer.v1.mjs';
const read=p=>JSON.parse(fs.readFileSync(new URL(p,import.meta.url),'utf8'));
const good=read('../../control-plane/external-adapters/fixtures/PROCUREMENT_COHERENT_PO_RECEIPT_INVOICE_PAYMENT_v1.json');
const bad=read('../../control-plane/external-adapters/fixtures/PROCUREMENT_LOCALLY_VALID_RECEIPT_INVOICE_IDENTITY_MISMATCH_v1.json');
const a=evaluateNeutralEvidence(normalizeProcurementProfile(good));
assert.equal(a.result,'EVIDENCE_ELIGIBLE_FOR_CLOSURE_EVALUATION');
assert.equal(a.primaryFailureClass,null);
const b=evaluateNeutralEvidence(normalizeProcurementProfile(bad));
assert.equal(b.result,'FAIL_CLOSED');
assert.equal(b.primaryFailureClass,'identity_conflict');
assert.deepEqual(b.consequentFailureClasses,['continuity_unproven','closure_held']);
assert.equal(b.createsAuthority,false);
process.stdout.write(JSON.stringify({schema:'GOVERNANCE_SUBSTRATE_NON_SOFTWARE_PROCUREMENT_SELF_TEST_RECEIPT_v1',result:'PASS',controls:[
 {fixture:good.fixtureId,result:a.result,primaryFailureClass:a.primaryFailureClass},
 {fixture:bad.fixtureId,result:b.result,primaryFailureClass:b.primaryFailureClass,consequences:b.consequentFailureClasses}
]},null,2)+'\n');
