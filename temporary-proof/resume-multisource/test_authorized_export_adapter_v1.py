#!/usr/bin/env python3
import unittest,sys
sys.path.insert(0,"tools/resume-corpus")
from authorized_export_adapter_v1 import process
def row(src,txt,permission="EXPLICIT_AUTHORIZED",us="true",authentic="true"):
 return {"source_id":src,"record_id":"fixture","permission":permission,"us_market":us,"authentic_resume":authentic,"normalized_resume_text":txt}
class Tests(unittest.TestCase):
 def test_multi_source_union(self):
  rows=[row("OPEN_DATASETS","Work Experience Engineer 2019 Education"),row("DIRECT_UPLOAD","Work Experience Engineer 2019 Education"),row("PARTNER_FEEDS","Work Experience Driver 2020 Education")]
  r=process(rows,{"OPEN_DATASETS","DIRECT_UPLOAD","PARTNER_FEEDS"})
  self.assertEqual(r["globalUnique"],2)
  self.assertEqual(r["sourceCounts"]["DIRECT_UPLOAD"]["duplicate"],1)
 def test_fail_closed(self):
  rows=[row("LINKEDIN","A"),row("OPEN_DATASETS","B",permission="UNKNOWN"),row("OPEN_DATASETS","C",authentic="false"),row("OPEN_DATASETS","D",us="false")]
  r=process(rows,{"OPEN_DATASETS"})
  self.assertEqual(r["globalUnique"],0)
  self.assertEqual(sum(r["rejectedReasons"].values()),4)
 def test_no_content_in_receipt(self):
  sensitive="Private Name private@example.com 555-222-3333 Work Experience"
  r=process([row("DIRECT_UPLOAD",sensitive)],{"DIRECT_UPLOAD"})
  self.assertNotIn("private@example.com",str(r))
  self.assertNotIn("Private Name",str(r))
if __name__=="__main__":unittest.main()
