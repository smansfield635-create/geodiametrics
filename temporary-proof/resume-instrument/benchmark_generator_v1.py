#!/usr/bin/env python3
"""Frozen non-PII adversarial benchmark generator for resume detector/dedup instrumentation.
Templates are independently labeled by construction. No real candidate data is included.
"""
import json, random, hashlib, re
SEED=20261009
rng=random.Random(SEED)
FIRST=["Alex","Jordan","Taylor","Morgan","Casey","Riley","Avery","Cameron","Quinn","Parker"]
LAST=["Example","Sample","Test","Control","Fixture"]
ROLES=["Sales Manager","Software Engineer","Truck Driver","Operations Analyst","Warehouse Coordinator","Project Manager","Electrician","Nurse","Accountant","Marketing Specialist","Mechanic","Teacher"]
SKILLS=["Excel","Python","Salesforce","forklift","SQL","project planning","customer service","inventory","AWS","welding","budgeting","training"]
def resume(i):
 n=f"{rng.choice(FIRST)} {rng.choice(LAST)}"
 role=rng.choice(ROLES); prior=rng.choice([x for x in ROLES if x!=role])
 y=2015+i%7; y2=y+3
 return f"""RESUME
{n}
Professional Summary
Experienced {role.lower()} with demonstrated operational experience.
Work Experience
{role} | Example Company | {y2} - Present
Responsible for {rng.choice(SKILLS)} and team results.
{prior} | Sample Organization | {y} - {y2}
Delivered measurable improvements using {rng.choice(SKILLS)}.
Education
State College | 2014
Skills
{rng.choice(SKILLS)}, {rng.choice(SKILLS)}, {rng.choice(SKILLS)}
"""
def negative(i):
 kinds=[
 lambda:f"""JOB DESCRIPTION
We are hiring a {rng.choice(ROLES)}.
Responsibilities include {rng.choice(SKILLS)} and team collaboration.
Requirements: experience preferred. Posted 2024. Apply today.""",
 lambda:f"""COMPANY BIOGRAPHY
Founded in {2000+i%20}, Example Company employs managers, engineers and specialists.
Our services include {rng.choice(SKILLS)} and consulting. Contact our office for information.""",
 lambda:f"""RESEARCH ARTICLE
Abstract: This study examines employment from {2018+i%4} to 2024.
Methods, results and discussion evaluate {rng.choice(SKILLS)} trends in the United States.""",
 lambda:f"""PROJECT PORTFOLIO
Project Alpha completed {2019+i%5}. Project Beta completed 2024.
Services: {rng.choice(SKILLS)}, design and implementation. This page describes organizational work.""",
 lambda:f"""TRAINING CATALOG
Course: {rng.choice(SKILLS)} Fundamentals.
Schedule 2023 - 2024. Instructor biography and course objectives. Certificate available after completion."""
 ]
 return kinds[i%len(kinds)]()
docs=[{"id":f"R{i:03d}","label":"RESUME","text":resume(i)} for i in range(120)]
docs += [{"id":f"N{i:03d}","label":"NOT_RESUME","text":negative(i)} for i in range(120)]
pairs=[]
for i in range(40):
 a=docs[i]["text"]; pairs.append({"id":f"E{i:03d}","label":"DUPLICATE","a":a,"b":a})
for i in range(30):
 a=docs[40+i]["text"]; b=a+"\nEmail: changed%d@example.invalid\nPhone: 214-555-%04d"%(i,1000+i); pairs.append({"id":f"C{i:03d}","label":"DUPLICATE","a":a,"b":b})
for i in range(30):
 a=docs[70+i]["text"]; b=a.upper().replace("\n","  \n\n"); pairs.append({"id":f"F{i:03d}","label":"DUPLICATE","a":a,"b":b})
for i in range(30):
 a=docs[i]["text"]; b=a+f"\nAdditional Experience\nDirector | New Employer | 2025 - Present\nSkills\nstrategy{i}"; pairs.append({"id":f"M{i:03d}","label":"DISTINCT","a":a,"b":b})
for i in range(30):
 pairs.append({"id":f"U{i:03d}","label":"DISTINCT","a":docs[i]["text"],"b":docs[90+i]["text"]})
out={"schema":"LBS_RESUME_INSTRUMENT_BENCHMARK_v1","seed":SEED,"labelsEstablishedBy":"GENERATOR_CONSTRUCTION_RULES_BEFORE_DETECTOR_EXECUTION","containsRealCandidateData":False,"documents":docs,"dedupPairs":pairs}
print(json.dumps(out,sort_keys=True,separators=(",",":")))
