import json,subprocess,sys,unittest
class T(unittest.TestCase):
 def test_queue(self):
  r=json.loads(subprocess.check_output([sys.executable,"tools/resume-corpus/rank_zero_cost_us_sources_v1.py"],text=True))
  self.assertGreaterEqual(r["leadCount"],8)
  self.assertEqual(r["countableNow"],0)
  self.assertEqual(r["nextGate"],"EXPLICIT_CORPUS_PROCESSING_PERMISSION")
  ids=[x["id"] for x in r["top"][:6]]
  self.assertIn("MIT_SLOAN_RESUME_BOOKS",ids)
  self.assertIn("CORNELL_ILR_LATE_SPRING_BOOK",ids)
if __name__=="__main__":unittest.main()
