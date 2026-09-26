import fs from 'node:fs';import {evaluateNeutralEvidenceV2} from './neutral-evaluator.v2.mjs';
const [p,o]=process.argv.slice(2);if(!p||!o)throw new Error('usage');
const packet=JSON.parse(fs.readFileSync(p,'utf8'));
const outputs=packet.cases.map(({domain,...input})=>({domain,...evaluateNeutralEvidenceV2(input)}));
const receipt={schema:'GGAQ_V2_BLIND_EXECUTION_OUTPUT',protocolBlob:packet.protocolBlob,evaluatorBlob:packet.evaluatorBlob,caseCount:outputs.length,outputs};
fs.writeFileSync(o,JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify({schema:receipt.schema,caseCount:outputs.length,results:outputs.map(x=>({evaluationId:x.evaluationId,result:x.result,primaryFailureClass:x.primaryFailureClass,consequentFailureClasses:x.consequentFailureClasses}))},null,2));