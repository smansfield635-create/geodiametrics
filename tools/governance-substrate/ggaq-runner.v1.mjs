import fs from 'node:fs';
import {evaluateNeutralEvidence} from './neutral-evaluator.v1.mjs';
const [packetPath,outPath]=process.argv.slice(2);
if(!packetPath||!outPath) throw new Error('usage: node ggaq-runner packet output');
const packet=JSON.parse(fs.readFileSync(packetPath,'utf8'));
const outputs=packet.cases.map(({domain,...input})=>({domain,...evaluateNeutralEvidence(input)}));
const receipt={schema:'GENERAL_GOVERNANCE_ADVERSARIAL_EXECUTION_OUTPUT_v1',protocol:packet.protocol,evaluatorBlob:packet.evaluatorBlob,caseCount:outputs.length,outputs};
fs.writeFileSync(outPath,JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify({schema:receipt.schema,caseCount:outputs.length,results:outputs.map(x=>({evaluationId:x.evaluationId,result:x.result,primaryFailureClass:x.primaryFailureClass,consequentFailureClasses:x.consequentFailureClasses}))},null,2));
