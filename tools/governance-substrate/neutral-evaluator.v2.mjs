export const RECEIPT_SCHEMA_V2='GOVERNANCE_SUBSTRATE_NON_AUTHORITATIVE_EVALUATION_RECEIPT_v2';
const E=new Set(['ESTABLISHED','NEGATED','UNPROVEN','CONFLICTING']);
const stable=v=>Array.isArray(v)?v.map(stable):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,stable(v[k])])):v;
const refs=(...g)=>[...new Set(g.flat().filter(Boolean))].sort();
const unknown=s=>s==='UNPROVEN'||s==='CONFLICTING';
const valid=s=>E.has(s);
export function evaluateNeutralEvidenceV2(x){
 for(const k of ['admission','authority','qualification','execution','materialization','correspondence','continuity','closure','subject']) if(!x?.[k]) throw new Error('INVALID_INPUT:'+k);
 for(const s of [x.admission.state,x.authority.state,x.qualification.state,x.correspondence.state,x.continuity.state,x.closure.evidenceCompletenessState]) if(!valid(s)) throw new Error('INVALID_EPISTEMIC_STATE:'+s);
 const preservedEvidence=refs(x.admission.evidenceRefs,x.authority.evidenceRefs,x.qualification.evidenceRefs,x.execution.evidenceRefs,x.materialization.evidenceRefs,x.correspondence.evidenceRefs,x.continuity.evidenceRefs);
 const identityConflict=x.correspondence.state==='NEGATED'&&x.correspondence.fromIdentity!==x.correspondence.toIdentity;
 const closureIdentityBound=x.closure.subjectIdentity===x.subject.closureSubjectIdentity&&x.closure.subjectIdentity===x.correspondence.toIdentity&&x.closure.subjectIdentity===x.materialization.outputIdentity;
 let primary=null,consequences=[],result='EVIDENCE_ELIGIBLE_FOR_CLOSURE_EVALUATION',terminalEffect='CLOSURE_ELIGIBILITY_ONLY_NO_AUTHORITY';
 const epistemicUnknown=[
  ['admission',x.admission.state],['authority',x.authority.state],['qualification',x.qualification.state],
  ['correspondence',x.correspondence.state],['continuity',x.continuity.state],['closureEvidence',x.closure.evidenceCompletenessState]
 ].filter(([,s])=>unknown(s));
 if(epistemicUnknown.length){primary='UNKNOWN_FAIL_CLOSED';consequences=['closure_held'];}
 else if(x.admission.state==='NEGATED'){primary='admission_refused';consequences=['closure_held'];}
 else if(x.authority.state==='NEGATED'){primary='authority_lost';consequences=['closure_held'];}
 else if(identityConflict){primary='identity_conflict';consequences=['continuity_unproven','closure_held'];}
 else if(x.execution.status==='SURFACE_UNAVAILABLE'){primary='surface_unavailable';consequences=['closure_held'];}
 else if(x.execution.status==='INTERRUPTED'){primary='execution_interrupted';consequences=['closure_held'];}
 else if(x.qualification.state==='NEGATED'){primary='product_evidence_failure';consequences=['closure_held'];}
 else if(x.correspondence.state==='NEGATED'||x.continuity.state==='NEGATED'){primary='continuity_unproven';consequences=['closure_held'];}
 else if(x.execution.status==='FAILURE'){primary='product_evidence_failure';consequences=['closure_held'];}
 else if(x.closure.evidenceCompletenessState==='NEGATED'||!closureIdentityBound||x.closure.requested!==true){primary='closure_held';}
 if(primary){result='FAIL_CLOSED';terminalEffect='CLOSURE_HELD_NO_TERMINAL_AUTHORITY';}
 return stable({schema:RECEIPT_SCHEMA_V2,evaluationId:x.evaluationId,result,primaryFailureClass:primary,consequentFailureClasses:consequences,terminalEffect,preservedEvidence,epistemicUnknown:epistemicUnknown.map(([field,state])=>({field,state})),createsAuthority:false});
}