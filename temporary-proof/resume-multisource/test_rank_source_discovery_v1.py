import json,subprocess,sys,unittest
class T(unittest.TestCase):
 def test_ranking(self):
  r=json.loads(subprocess.check_output([sys.executable,"tools/resume-corpus/rank_source_discovery_v1.py"],text=True))
  self.assertEqual(r["countableNow"],0)
  ids=[x["id"] for x in r["advance"]]
  self.assertEqual(ids[0],"HANDSHAKE_EMPLOYER_PUBLIC_RESUMES")
  self.assertIn("UNIVERSITY_RESUME_BOOK_NETWORK",ids)
  killed={x["id"] for x in r["kill"]}
  self.assertIn("RESUME_SCREENING_10K",killed)
  self.assertIn("JOBHOP_VDAB",killed)
if __name__=="__main__":unittest.main()
