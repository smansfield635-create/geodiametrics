#!/usr/bin/env python3
import json,sys
sys.path.insert(0,"tools/resume-corpus")
from multisource_orchestrator_v1 import GlobalLedger,validate_registry,ranked_registry
reg=json.load(open("control-plane/resume-corpus/source-registry.v1.json"))
assert validate_registry(reg)
l=GlobalLedger()
assert l.ingest("CC_WEB_INDEX","Resume Alpha\nSales Manager 2018 - Present\nEducation State College")== "ACCEPT"
assert l.ingest("OPEN_DATASETS","Resume Beta\nSoftware Engineer 2020 - Present\nEducation State College")== "ACCEPT"
assert l.ingest("PUBLIC_DOCUMENT_WEB","Resume Alpha\nSales Manager 2018 - Present\nEducation State College")== "DUPLICATE"
assert l.ingest("UNIVERSITY_PORTFOLIO","Resume Gamma\nTruck Driver 2016 - Present\nCDL Education")== "ACCEPT"
assert l.ingest("FACEBOOK","Profile only",is_resume=False,authorized=False)== "REJECT"
r=l.receipt();assert r["globalUnique"]==3
assert sum(x["duplicates"] for x in r["sources"])==1
assert sum(x["rejected"] for x in r["sources"])==1
ranked=ranked_registry(reg,l);assert ranked[0]["status"]=="DISCOVERY_ACTIVE"
print(json.dumps({"schema":"LBS_MULTISOURCE_ORCHESTRATOR_SELF_TEST_v2","result":"PASS_CLOSED","registeredSources":len(reg["sources"]),"globalUniqueFixture":3,"crossSourceDuplicatesCollapsed":1,"unauthorizedOrNonResumeExcluded":1},sort_keys=True))
