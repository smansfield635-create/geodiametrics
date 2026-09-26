import fs from 'node:fs';
import {evaluateNeutralEvidence} from './neutral-evaluator.v1.mjs';

export function normalizeProcurementProfile(f) {
  const match=f.receiving.receivedSubjectIdentity===f.invoice.billedSubjectIdentity;
  return {
    evaluationId:f.fixtureId,
    subject:{candidateIdentity:f.purchaseOrder.orderedSubjectIdentity,closureSubjectIdentity:f.paymentRequest.subjectIdentity},
    admission:{admitted:f.purchaseOrder.status==='APPROVED'&&f.approval.decision==='APPROVED',evidenceRefs:[f.purchaseOrder.evidenceId,f.approval.evidenceId]},
    authority:{established:Boolean(f.approval.approverIdentity&&f.approval.scope),evidenceRefs:[f.approval.evidenceId]},
    qualification:{subjectIdentity:f.receiving.receivedSubjectIdentity,passed:f.receiving.inspectionResult==='PASS',evidenceRefs:[f.receiving.evidenceId]},
    execution:{status:'SUCCESS',inputIdentity:f.receiving.receivedSubjectIdentity,outputIdentity:f.invoice.billedSubjectIdentity,evidenceRefs:[f.receiving.evidenceId,f.invoice.evidenceId]},
    materialization:{outputIdentity:f.invoice.billedSubjectIdentity,evidenceRefs:[f.invoice.evidenceId]},
    correspondence:{fromIdentity:f.receiving.receivedSubjectIdentity,toIdentity:f.invoice.billedSubjectIdentity,established:match,evidenceRefs:match?[f.receiving.evidenceId,f.invoice.evidenceId]:[]},
    continuity:{established:match,evidenceRefs:match?[f.receiving.evidenceId,f.invoice.evidenceId]:[]},
    closure:{requested:f.paymentRequest.requested===true,subjectIdentity:f.paymentRequest.subjectIdentity,requiredEvidencePresent:Array.isArray(f.audit.references)&&f.audit.references.length>=4}
  };
}

function main(){
 const [inputPath,outputPath]=process.argv.slice(2);
 if(!inputPath||!outputPath) throw new Error('usage: node procurement-normalizer input.json output.json');
 const fixture=JSON.parse(fs.readFileSync(inputPath,'utf8'));
 const receipt=evaluateNeutralEvidence(normalizeProcurementProfile(fixture));
 fs.writeFileSync(outputPath,JSON.stringify(receipt,null,2)+'\n');
}
if(process.argv[1]&&import.meta.url===new URL('file://'+process.argv[1]).href) main();
