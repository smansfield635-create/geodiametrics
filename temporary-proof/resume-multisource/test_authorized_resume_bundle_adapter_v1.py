#!/usr/bin/env python3
import io,json,pathlib,sqlite3,tempfile,unittest,zipfile,sys
sys.path.insert(0,"tools/resume-corpus")
from authorized_resume_bundle_adapter_v1 import process

R1="""PROFESSIONAL SUMMARY Sales manager
WORK EXPERIENCE
Sales Manager 2019 - Present
Account Representative 2016 - 2019
EDUCATION State University 2016
SKILLS CRM leadership forecasting
Email: person@example.com
Phone: 555-555-1234
"""
R2="""SUMMARY Truck driver
WORK EXPERIENCE
Truck Driver 2018 - Present
Operator 2015 - 2018
EDUCATION Community College 2014
SKILLS CDL logistics safety
"""
BAD="This is a conference agenda and speaker schedule."

def authority(source):
 return {"schema":"LBS_SOURCE_AUTHORITY_ASSERTION_v1","sourceId":source,"authorityState":"EXPLICIT_AUTHORIZED","usMarketState":"VERIFIED_US_MARKET_CHANNEL","documentPopulationState":"VERIFIED_HUMAN_SUBMITTED_RESUMES","authorizationBasis":"fixture authorization","authenticityBasis":"fixture represents human-submitted export"}

class T(unittest.TestCase):
 def test_zip_cross_source_global_dedup(self):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);db=d/"ledger.sqlite"
   for src,docs in [("HANDSHAKE_EMPLOYER_EXPORT",[R1,R2]),("UNIVERSITY_OPT_IN_RESUME_BOOK",[R1,BAD])]:
    z=d/(src+".zip")
    with zipfile.ZipFile(z,"w") as w:
     for i,t in enumerate(docs):w.writestr(f"{i}.txt",t)
    r=process(src,z,authority(src),db)
   self.assertEqual(r["globalUnique"],2)
   self.assertEqual(r["duplicates"],1)
   self.assertEqual(r["rejected"],1)
   self.assertFalse(r["rawResumeContentPersisted"])
 def test_fails_closed_without_authority(self):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);p=d/"r.txt";p.write_text(R1)
   a=authority("HANDSHAKE_EMPLOYER_EXPORT");a["authorityState"]="UNKNOWN"
   with self.assertRaises(PermissionError):process("HANDSHAKE_EMPLOYER_EXPORT",p,a,d/"l.sqlite")
 def test_rejects_unknown_authenticity(self):\n  with tempfile.TemporaryDirectory() as d:\n   d=pathlib.Path(d);p=d/"r.txt";p.write_text(R1)\n   a=authority("HANDSHAKE_EMPLOYER_EXPORT");a["documentPopulationState"]="UNKNOWN"\n   with self.assertRaises(PermissionError):process("HANDSHAKE_EMPLOYER_EXPORT",p,a,d/"l.sqlite")\n def test_ledger_has_hashes_not_resume_text(self):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);p=d/"r.txt";p.write_text(R1);db=d/"l.sqlite"
   process("HANDSHAKE_EMPLOYER_EXPORT",p,authority("HANDSHAKE_EMPLOYER_EXPORT"),db)
   con=sqlite3.connect(db)
   rows=str(con.execute("select * from resumes").fetchall())
   self.assertNotIn("person@example.com",rows);self.assertNotIn("Sales Manager",rows)
if __name__=="__main__":unittest.main()
