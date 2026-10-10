#!/usr/bin/env python3
import json,subprocess,sys,tempfile,pathlib,unittest
class T(unittest.TestCase):
 def test_zero_owner_actions(self):
  d={"sources":[
   {"sourceId":"A","state":"DISCOVERED"},
   {"sourceId":"B","state":"Q5_PROJECTED","I":2000000,"r":0.95,"u":0.9,"d":0.8,"p":0.9},
   {"sourceId":"C","state":"Q6_RIGHTS_GATE","rightsGate":"PASS"},
   {"sourceId":"D","state":"Q7_MILLION_CAPABLE"}
  ]}
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/"s.json";p.write_text(json.dumps(d))
   r=json.loads(subprocess.check_output([sys.executable,"tools/resume-corpus/autonomous_source_coordinator_v2.py",str(p)],text=True))
   self.assertFalse(r["ownerActionRequired"])
   self.assertEqual(r["ownerActionCount"],0)
   self.assertEqual(r["systemActionCount"],4)
   self.assertEqual(r["sources"][1]["nextAction"],"ADVANCE_Q6_RIGHTS_GATE")
 def test_submillion_projection_not_promoted(self):
  d={"sources":[{"sourceId":"A","state":"Q5_PROJECTED","I":1000000,"r":0.5,"u":0.5,"d":0.5,"p":1.0}]}
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td)/"s.json";p.write_text(json.dumps(d))
   r=json.loads(subprocess.check_output([sys.executable,"tools/resume-corpus/autonomous_source_coordinator_v2.py",str(p)],text=True))
   self.assertEqual(r["sources"][0]["nextAction"],"KILL_OR_COMBINE_SOURCE")
if __name__=="__main__":unittest.main()
