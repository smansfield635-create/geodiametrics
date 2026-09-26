import fs from 'node:fs';

export const RECEIPT_SCHEMA = 'GOVERNANCE_SUBSTRATE_NON_AUTHORITATIVE_EVALUATION_RECEIPT_v1';

const stable = value => Array.isArray(value) ? value.map(stable) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(k => [k, stable(value[k])])) : value;
const refs = (...groups) => [...new Set(groups.flat().filter(Boolean))].sort();

function requireObject(value, field) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Object.assign(new Error('INVALID_INPUT:'+field), {code:'INVALID_INPUT',field});
  return value;
}
function requireString(value, field) {
  if (typeof value !== 'string' || !value) throw Object.assign(new Error('INVALID_INPUT:'+field), {code:'INVALID_INPUT',field});
  return value;
}

export function evaluateNeutralEvidence(input) {
  const x=requireObject(input,'input');
  const evaluationId=requireString(x.evaluationId,'evaluationId');
  const subject=requireObject(x.subject,'subject');
  const admission=requireObject(x.admission,'admission');
  const authority=requireObject(x.authority,'authority');
  const qualification=requireObject(x.qualification,'qualification');
  const execution=requireObject(x.execution,'execution');
  const materialization=requireObject(x.materialization,'materialization');
  const correspondence=requireObject(x.correspondence,'correspondence');
  const continuity=requireObject(x.continuity,'continuity');
  const closure=requireObject(x.closure,'closure');

  const preservedEvidence=refs(admission.evidenceRefs,authority.evidenceRefs,qualification.evidenceRefs,execution.evidenceRefs,materialization.evidenceRefs,correspondence.evidenceRefs,continuity.evidenceRefs);
  const invariantResults=[];

  const push=(invariant,pass,detail)=>invariantResults.push({invariant,result:pass?'PASS':'FAIL',detail});

  const identitiesExplicit=[
    subject.candidateIdentity,subject.closureSubjectIdentity,qualification.subjectIdentity,
    execution.inputIdentity,execution.outputIdentity,materialization.outputIdentity,
    correspondence.fromIdentity,correspondence.toIdentity,closure.subjectIdentity
  ].every(v=>typeof v==='string'&&v.length>0);
  push('IDENTITY_MUST_BE_EXPLICIT_AT_EACH_CLAIM_BOUNDARY',identitiesExplicit,identitiesExplicit?'all required identities explicit':'one or more required identities absent');

  push('AUTHORITY_IS_NOT_INFERRED_FROM_EXECUTION_OR_PRESENTATION',authority.established===true,'authority evaluated independently of execution/workflow state');
  push('ADMISSION_IS_NOT_SUCCESS',admission.admitted===true,'admission is an independent prerequisite, not evidence of successful execution');

  const qualificationBound=qualification.passed!==true || qualification.subjectIdentity===correspondence.fromIdentity;
  push('QUALIFICATION_REQUIRES_EVIDENCE_BOUND_TO_THE_QUALIFIED_SUBJECT',qualificationBound,qualificationBound?'qualification subject is bound to correspondence source':'qualification subject differs from correspondence source');

  const materializationBound=materialization.outputIdentity===execution.outputIdentity;
  push('MATERIALIZED_OUTPUT_MUST_RETAIN_PROVENANCE_TO_EXECUTION',materializationBound,materializationBound?'materialized output matches execution output':'materialized output differs from execution output');

  const identityConflict=correspondence.fromIdentity!==correspondence.toIdentity && correspondence.established!==true;
  const transitionProven=correspondence.established===true && continuity.established===true;
  push('LOCAL_CORRECTNESS_DOES_NOT_PROVE_TRANSITION_CORRESPONDENCE',!identityConflict || !transitionProven,'local success does not override unproven/different transition identities');
  push('CONTINUITY_REQUIRES_TRANSITION_EVIDENCE',transitionProven,'continuity requires established correspondence and continuity evidence');

  const closureIdentityBound=closure.subjectIdentity===subject.closureSubjectIdentity && closure.subjectIdentity===correspondence.toIdentity && closure.subjectIdentity===materialization.outputIdentity;
  const closureReady=closure.requested===true && closure.requiredEvidencePresent===true && closureIdentityBound && transitionProven && qualification.passed===true && execution.status==='SUCCESS';
  push('CLOSURE_REQUIRES_DOMAIN_APPROPRIATE_EXACT_CORRESPONDENCE_PROVENANCE_LINEAGE_AND_REREAD_OR_VERIFICATION',closureReady,closureReady?'closure prerequisites established':'one or more closure prerequisites unestablished');

  let primary=null, consequences=[], terminalEffect='CLOSURE_ELIGIBILITY_ONLY_NO_AUTHORITY', result='EVIDENCE_ELIGIBLE_FOR_CLOSURE_EVALUATION';

  if (admission.admitted!==true) {
    primary='admission_refused'; consequences=['closure_held'];
  } else if (authority.established!==true) {
    primary='authority_lost'; consequences=['closure_held'];
  } else if (identityConflict) {
    primary='identity_conflict'; consequences=['continuity_unproven','closure_held'];
  } else if (execution.status==='SURFACE_UNAVAILABLE') {
    primary='surface_unavailable'; consequences=['closure_held'];
  } else if (execution.status==='INTERRUPTED') {
    primary='execution_interrupted'; consequences=['closure_held'];
  } else if (qualification.passed===false) {
    primary='product_evidence_failure'; consequences=['closure_held'];
  } else if (!transitionProven) {
    primary='continuity_unproven'; consequences=['closure_held'];
  } else if (!closureReady) {
    primary='closure_held'; consequences=[];
  }

  if (primary) {
    result='FAIL_CLOSED';
    terminalEffect='CLOSURE_HELD_NO_TERMINAL_AUTHORITY';
  }

  return stable({
    schema:RECEIPT_SCHEMA,
    evaluationId,
    result,
    invariantResults,
    primaryFailureClass:primary,
    consequentFailureClasses:consequences,
    terminalEffect,
    preservedEvidence,
    createsAuthority:false
  });
}

export function normalizeJiraScrumProfile(fixture) {
  const f=requireObject(fixture,'fixture');
  return stable({
    evaluationId:f.fixtureId,
    subject:{candidateIdentity:f.implementation.commitIdentity,closureSubjectIdentity:f.closureRequest.subjectIdentity},
    admission:{admitted:f.workflow.decision==='ALLOWED'&&f.approval.decision==='APPROVED',evidenceRefs:[f.workflow.evidenceId,f.approval.evidenceId]},
    authority:{established:Boolean(f.authorityContext.role&&f.authorityContext.scope),evidenceRefs:[f.authorityContext.evidenceId]},
    qualification:{subjectIdentity:f.build.artifactIdentity,passed:f.build.testResult==='PASS'&&f.build.buildStatus==='SUCCESS'&&f.review.reviewState==='APPROVED'&&f.review.pullRequestState==='MERGED',evidenceRefs:[f.build.evidenceId,f.review.evidenceId]},
    execution:{status:f.deployment.status==='SUCCESS'?'SUCCESS':f.deployment.status,inputIdentity:f.build.artifactIdentity,outputIdentity:f.deployment.artifactIdentity,evidenceRefs:[f.deployment.evidenceId]},
    materialization:{outputIdentity:f.deployment.artifactIdentity,evidenceRefs:[f.deployment.evidenceId]},
    correspondence:{fromIdentity:f.build.artifactIdentity,toIdentity:f.deployment.artifactIdentity,established:f.build.artifactIdentity===f.deployment.artifactIdentity,evidenceRefs:f.build.artifactIdentity===f.deployment.artifactIdentity?[f.build.evidenceId,f.deployment.evidenceId]:[]},
    continuity:{established:f.build.artifactIdentity===f.deployment.artifactIdentity,evidenceRefs:f.build.artifactIdentity===f.deployment.artifactIdentity?[f.build.evidenceId,f.deployment.evidenceId]:[]},
    closure:{requested:f.closureRequest.requested===true,subjectIdentity:f.closureRequest.subjectIdentity,requiredEvidencePresent:Array.isArray(f.audit.references)&&f.audit.references.length>=7}
  });
}

function main(){
 const [fixturePath,outputPath]=process.argv.slice(2);
 if(!fixturePath||!outputPath) throw new Error('usage: node evaluator fixture.json output.json');
 const fixture=JSON.parse(fs.readFileSync(fixturePath,'utf8'));
 const receipt=evaluateNeutralEvidence(normalizeJiraScrumProfile(fixture));
 fs.writeFileSync(outputPath,JSON.stringify(receipt,null,2)+'\n');
}

if(process.argv[1]&&import.meta.url===new URL('file://'+process.argv[1]).href) main();
