import {evaluateNeutralEvidence} from './neutral-evaluator.v1.mjs';
export function normalizeRegulatedDocumentProfile(f){
 const same=f.qualification.subjectIdentity===f.issuance.issuedIdentity&&f.delivery.subjectIdentity===f.issuance.issuedIdentity;
 return {
  evaluationId:f.fixtureId,
  subject:{candidateIdentity:f.document.identity,closureSubjectIdentity:f.delivery.subjectIdentity},
  admission:{admitted:f.admission.decision==='ADMITTED',evidenceRefs:[f.admission.evidenceId]},
  authority:{established:f.approvalAuthority.approvalStatus==='APPROVED'&&f.approvalAuthority.currentAuthorityStatus==='ACTIVE',evidenceRefs:[f.approvalAuthority.approvalEvidenceId,f.approvalAuthority.currentAuthorityEvidenceId]},
  qualification:{subjectIdentity:f.qualification.subjectIdentity,passed:f.qualification.reviewResult==='PASS',evidenceRefs:[f.qualification.evidenceId]},
  execution:{status:f.issuance.status,inputIdentity:f.issuance.inputIdentity,outputIdentity:f.issuance.issuedIdentity,evidenceRefs:[f.issuance.evidenceId]},
  materialization:{outputIdentity:f.issuance.issuedIdentity,evidenceRefs:[f.issuance.evidenceId]},
  correspondence:{fromIdentity:f.qualification.subjectIdentity,toIdentity:f.issuance.issuedIdentity,established:same,evidenceRefs:same?[f.qualification.evidenceId,f.issuance.evidenceId]:[]},
  continuity:{established:same&&f.delivery.acknowledged===true,evidenceRefs:same&&f.delivery.acknowledged?[f.issuance.evidenceId,f.delivery.evidenceId]:[]},
  closure:{requested:true,subjectIdentity:f.delivery.subjectIdentity,requiredEvidencePresent:Array.isArray(f.audit.references)&&f.audit.references.length>=7}
 };
}
