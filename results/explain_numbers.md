# Explanation numbers (written by src/explain.py)

Train set: 105,145 rows, 7,080 people. Final model: CatBoost + calibration. Percentages are shares of rows (one row per person per survey round), unweighted.

## Partnered share in the train set, by sex and age

| sex | 18-24 | 25-29 | 30-34 | 35-43 |
|---|---|---|---|---|
| female | 0.303 | 0.555 | 0.618 | 0.626 |
| male | 0.19 | 0.473 | 0.588 | 0.634 |

## Importance: average push on a person's probability (percentage points, TreeSHAP)

How far, on average, each feature moves one person's predicted probability up or down from the average prediction. `rank` in the final model; `fold ranks` the range of its rank across the 5 fold models.

| feature | all | male | female | rank | fold ranks |
|---|---|---|---|---|---|
| age | 8.9 | 8.6 | 9.3 | 1 | 1-1 |
| race_ethnicity | 6.5 | 4.7 | 8.4 | 2 | 2-2 |
| earnings | 5.7 | 6.9 | 4.5 | 3 | 3-3 |
| sex | 4.8 | 4.5 | 5.1 | 4 | 4-4 |
| enrolled | 3.7 | 3.0 | 4.3 | 5 | 5-5 |
| bmi | 1.9 | 1.9 | 2.0 | 6 | 6-6 |
| big5_extraversion | 1.9 | 1.7 | 2.0 | 7 | 7-10 |
| big5_conscientiousness | 1.5 | 1.4 | 1.6 | 8 | 7-9 |
| census_region | 1.4 | 1.3 | 1.5 | 9 | 8-11 |
| mother_educ_grade | 1.4 | 1.5 | 1.2 | 10 | 7-13 |
| education | 1.3 | 1.2 | 1.3 | 11 | 8-13 |
| hours_worked | 1.2 | 1.7 | 0.7 | 12 | 9-15 |
| msa | 1.0 | 0.9 | 1.1 | 13 | 10-15 |
| family_income_1997 | 0.9 | 0.8 | 0.9 | 14 | 13-17 |
| religion | 0.8 | 0.8 | 0.9 | 15 | 14-18 |
| asvab_percentile | 0.8 | 0.7 | 0.8 | 16 | 14-17 |
| urban | 0.7 | 0.6 | 0.8 | 17 | 16-22 |
| nonresident_children | 0.7 | 0.9 | 0.5 | 18 | 14-22 |
| big5_emotional_stability | 0.7 | 0.7 | 0.7 | 19 | 14-21 |
| attendance_worship | 0.6 | 0.7 | 0.5 | 20 | 19-24 |
| height_cm | 0.6 | 0.6 | 0.6 | 21 | 18-25 |
| big5_openness | 0.6 | 0.5 | 0.6 | 22 | 18-27 |
| big5_agreeableness | 0.5 | 0.5 | 0.6 | 23 | 20-24 |
| lived_with_both_parents_age12 | 0.5 | 0.6 | 0.5 | 24 | 17-26 |
| father_educ_grade | 0.5 | 0.5 | 0.6 | 25 | 22-27 |
| weeks_worked | 0.5 | 0.3 | 0.6 | 26 | 23-26 |
| general_health | 0.4 | 0.4 | 0.4 | 27 | 25-28 |
| importance_faith | 0.2 | 0.2 | 0.2 | 28 | 27-28 |

## Change one thing, averaged over real people (percentage points)

For up to 2,000 real people of each sex in the train set (those whose answer to this question is known), set the feature to 'from' and then to 'to', keep everything else as it is, and average the model's probability. `at 'from'`: the average probability at the 'from' value. `change`: final model, then (lowest to highest) across the 5 fold models, then stable / unstable / small (under 1 point).

| feature | male: from → to | male: at 'from' | male: change, points | female: from → to | female: at 'from' | female: change, points |
|---|---|---|---|---|---|---|
| age | 22 → 32 | 0.295 | +22.0 (+21.5 to +22.0) stable | 22 → 32 | 0.400 | +19.0 (+18.4 to +19.2) stable |
| earnings | 20,000 → 60,000 | 0.426 | +11.3 (+10.4 to +12.2) stable | 20,000 → 60,000 | 0.498 | +3.5 (+3.3 to +4.7) stable |
| weeks_worked | 0 → 52 | 0.402 | +0.1 (-0.4 to +0.5) small | 0 → 52 | 0.482 | -2.4 (-2.6 to -1.8) stable |
| hours_worked | 1,000 → 2,080 | 0.388 | +3.3 (+2.8 to +3.8) stable | 1,000 → 2,080 | 0.485 | -0.1 (-0.4 to +0.3) small |
| height_cm | 170 → 185 | 0.397 | +1.2 (+0.6 to +1.7) stable | 157 → 170 | 0.476 | -0.9 (-1.4 to -0.4) small |
| bmi | 22 → 32 | 0.379 | +4.8 (+4.5 to +5.6) stable | 22 → 32 | 0.451 | +3.8 (+3.6 to +4.6) stable |
| general_health | 1 → 4 | 0.404 | -0.5 (-1.4 to +0.0) small | 1 → 4 | 0.466 | -1.3 (-1.5 to -0.9) stable |
| asvab_percentile | 25 → 75 | 0.393 | +1.1 (+0.7 to +2.2) stable | 25 → 75 | 0.461 | +2.1 (+1.9 to +2.7) stable |
| mother_educ_grade | 12 → 16 | 0.416 | -4.5 (-5.5 to -2.6) stable | 12 → 16 | 0.482 | -2.0 (-2.8 to -0.4) stable |
| father_educ_grade | 12 → 16 | 0.406 | -1.1 (-1.8 to -0.5) stable | 12 → 16 | 0.478 | -1.3 (-2.2 to -0.8) stable |
| lived_with_both_parents_age12 | 0 → 1 | 0.404 | -1.2 (-1.7 to -0.5) stable | 0 → 1 | 0.487 | -0.3 (-0.7 to +0.1) small |
| family_income_1997 | 25,000 → 75,000 | 0.410 | -0.6 (-1.4 to -0.3) small | 25,000 → 75,000 | 0.482 | -0.1 (-1.2 to +0.6) small |
| urban | 0 → 1 | 0.431 | -1.8 (-2.2 to -1.2) stable | 0 → 1 | 0.497 | -3.2 (-3.3 to -2.5) stable |
| attendance_worship | 1 → 6 | 0.415 | +3.2 (+2.4 to +3.9) stable | 1 → 6 | 0.476 | +0.9 (+0.8 to +1.3) small |
| importance_faith | 5 → 1 | 0.565 | +0.6 (+0.7 to +1.4) small | 5 → 1 | 0.601 | +0.6 (+0.6 to +1.1) small |
| big5_extraversion | 3 → 6 | 0.547 | +5.5 (+4.1 to +6.1) stable | 3 → 6 | 0.574 | +4.9 (+3.8 to +5.2) stable |
| big5_agreeableness | 3 → 6 | 0.565 | +0.1 (-0.2 to +0.4) small | 3 → 6 | 0.590 | +0.2 (-0.2 to +0.8) small |
| big5_conscientiousness | 3 → 6 | 0.514 | +4.9 (+3.9 to +5.3) stable | 3 → 6 | 0.558 | +5.4 (+4.2 to +6.7) stable |
| big5_emotional_stability | 3 → 6 | 0.541 | +3.3 (+2.5 to +4.5) stable | 3 → 6 | 0.581 | +3.7 (+3.1 to +5.4) stable |
| big5_openness | 3 → 6 | 0.567 | +0.5 (+0.3 to +0.9) small | 3 → 6 | 0.608 | -0.7 (-1.0 to +0.3) small |
| nonresident_children | 0 → 1 | 0.408 | -3.4 (-4.0 to -2.2) stable | 0 → 1 | 0.474 | -2.0 (-2.5 to -1.0) stable |

## Categories: each value against the most common one (percentage points)

Same method as above.

| feature | value | compared with | sex | change, points |
|---|---|---|---|---|
| race_ethnicity | black | other | male | -9.6 (-10.1 to -9.0) stable |
| race_ethnicity | hispanic | other | male | -1.7 (-2.0 to -1.0) stable |
| race_ethnicity | black | other | female | -26.3 (-27.4 to -25.6) stable |
| race_ethnicity | hispanic | other | female | -1.6 (-2.4 to -1.3) stable |
| education | bachelor_plus | some_college | male | -2.1 (-3.0 to -0.7) stable |
| education | hs | some_college | male | +0.7 (+0.2 to +2.2) small |
| education | less_than_hs | some_college | male | +5.5 (+4.6 to +7.0) stable |
| education | bachelor_plus | some_college | female | -2.0 (-2.8 to -1.0) stable |
| education | hs | some_college | female | +0.6 (+0.2 to +1.4) small |
| education | less_than_hs | some_college | female | +5.1 (+4.4 to +5.9) stable |
| enrolled | college_2yr | not_enrolled | male | -4.3 (-4.7 to -3.6) stable |
| enrolled | college_4yr | not_enrolled | male | -5.5 (-6.1 to -5.0) stable |
| enrolled | graduate | not_enrolled | male | -3.7 (-5.2 to -1.8) stable |
| enrolled | high_school | not_enrolled | male | -5.5 (-6.1 to -4.8) stable |
| enrolled | college_2yr | not_enrolled | female | -7.9 (-8.4 to -7.0) stable |
| enrolled | college_4yr | not_enrolled | female | -10.2 (-10.6 to -9.5) stable |
| enrolled | graduate | not_enrolled | female | -6.8 (-8.4 to -3.9) stable |
| enrolled | high_school | not_enrolled | female | -10.2 (-10.6 to -9.5) stable |
| census_region | midwest | south | male | +0.4 (+0.1 to +1.0) small |
| census_region | northeast | south | male | -4.6 (-5.1 to -4.1) stable |
| census_region | west | south | male | -0.3 (-0.5 to +0.4) small |
| census_region | midwest | south | female | +0.1 (-0.5 to +0.3) small |
| census_region | northeast | south | female | -5.9 (-6.0 to -5.0) stable |
| census_region | west | south | female | -0.7 (-0.5 to -0.3) small |
| msa | cbsa_central_city | cbsa_not_central_city | male | -1.9 (-1.9 to -1.6) stable |
| msa | cbsa_not_known | cbsa_not_central_city | male | -1.3 (-1.5 to +0.1) unstable |
| msa | not_in_cbsa | cbsa_not_central_city | male | -0.4 (-1.1 to +0.2) small |
| msa | cbsa_central_city | cbsa_not_central_city | female | -2.4 (-3.0 to -2.0) stable |
| msa | cbsa_not_known | cbsa_not_central_city | female | -1.8 (-2.2 to +0.3) unstable |
| msa | not_in_cbsa | cbsa_not_central_city | female | -0.6 (-1.6 to +0.8) small |
| religion | catholic | protestant | male | -1.8 (-2.3 to -1.4) stable |
| religion | jewish | protestant | male | -1.8 (-2.5 to -1.0) stable |
| religion | none | protestant | male | -1.6 (-1.7 to -1.3) stable |
| religion | other | protestant | male | -1.5 (-2.2 to -0.8) stable |
| religion | catholic | protestant | female | -2.1 (-2.7 to -1.7) stable |
| religion | jewish | protestant | female | -2.0 (-2.1 to -1.3) stable |
| religion | none | protestant | female | -1.9 (-2.1 to -1.6) stable |
| religion | other | protestant | female | -1.1 (-1.6 to +0.2) unstable |

## Interactions: the same change by age band and sex (percentage points)

Same method, with the real people drawn from one age band at a time.

| change | sex | age | at 'from' | change, points |
|---|---|---|---|---|
| earnings $20k → $60k | male | 18-24 | 0.261 | +10.0 (+9.8 to +11.4) stable |
| earnings $20k → $60k | male | 25-29 | 0.482 | +12.3 (+11.2 to +13.2) stable |
| earnings $20k → $60k | male | 30-34 | 0.568 | +12.3 (+10.8 to +12.7) stable |
| earnings $20k → $60k | male | 35-43 | 0.582 | +12.1 (+10.5 to +12.6) stable |
| earnings $20k → $60k | female | 18-24 | 0.366 | +4.2 (+4.1 to +5.9) stable |
| earnings $20k → $60k | female | 25-29 | 0.560 | +2.5 (+2.2 to +3.8) stable |
| earnings $20k → $60k | female | 30-34 | 0.611 | +3.3 (+2.6 to +4.4) stable |
| earnings $20k → $60k | female | 35-43 | 0.618 | +3.4 (+2.3 to +4.5) stable |
| no job → full-time all year at $40k | male | 18-24 | 0.131 | +21.8 (+20.2 to +22.7) stable |
| no job → full-time all year at $40k | male | 25-29 | 0.333 | +24.3 (+22.7 to +25.9) stable |
| no job → full-time all year at $40k | male | 30-34 | 0.423 | +23.2 (+22.3 to +24.9) stable |
| no job → full-time all year at $40k | male | 35-43 | 0.439 | +22.6 (+21.9 to +23.9) stable |
| no job → full-time all year at $40k | female | 18-24 | 0.308 | +9.5 (+8.3 to +10.9) stable |
| no job → full-time all year at $40k | female | 25-29 | 0.583 | -1.7 (-1.5 to -0.9) stable |
| no job → full-time all year at $40k | female | 30-34 | 0.661 | -3.3 (-3.4 to -2.2) stable |
| no job → full-time all year at $40k | female | 35-43 | 0.659 | -2.9 (-3.6 to -2.1) stable |
| high school → bachelor's | male | 18-24 | 0.189 | -4.0 (-4.9 to -3.3) stable |
| high school → bachelor's | male | 25-29 | 0.483 | -6.3 (-8.5 to -5.1) stable |
| high school → bachelor's | male | 30-34 | 0.578 | +0.1 (-1.6 to +1.1) small |
| high school → bachelor's | male | 35-43 | 0.621 | +2.5 (+1.3 to +3.9) stable |
| high school → bachelor's | female | 18-24 | 0.292 | -5.8 (-7.0 to -5.3) stable |
| high school → bachelor's | female | 25-29 | 0.553 | -5.6 (-7.6 to -4.6) stable |
| high school → bachelor's | female | 30-34 | 0.609 | +2.1 (+0.6 to +3.2) stable |
| high school → bachelor's | female | 35-43 | 0.610 | +4.7 (+3.8 to +5.0) stable |
| worship never → weekly | male | 18-24 | 0.206 | +0.5 (+0.2 to +1.3) small |
| worship never → weekly | male | 25-29 | 0.473 | +4.2 (+3.1 to +5.4) stable |
| worship never → weekly | male | 30-34 | 0.565 | +5.6 (+4.4 to +5.8) stable |
| worship never → weekly | male | 35-43 | 0.612 | +5.5 (+4.3 to +5.7) stable |
| worship never → weekly | female | 18-24 | 0.323 | -1.1 (-1.3 to -0.5) stable |
| worship never → weekly | female | 25-29 | 0.541 | +1.3 (+1.1 to +2.1) stable |
| worship never → weekly | female | 30-34 | 0.612 | +2.8 (+2.0 to +3.1) stable |
| worship never → weekly | female | 35-43 | 0.623 | +2.8 (+1.9 to +2.9) stable |
| no → one child living elsewhere | male | 18-24 | 0.190 | -0.2 (-0.4 to +0.3) small |
| no → one child living elsewhere | male | 25-29 | 0.493 | -4.6 (-5.8 to -2.8) stable |
| no → one child living elsewhere | male | 30-34 | 0.602 | -6.1 (-7.0 to -4.0) stable |
| no → one child living elsewhere | male | 35-43 | 0.650 | -6.7 (-7.4 to -4.5) stable |
| no → one child living elsewhere | female | 18-24 | 0.302 | +1.9 (+1.3 to +2.6) stable |
| no → one child living elsewhere | female | 25-29 | 0.554 | -3.5 (-4.0 to -1.7) stable |
| no → one child living elsewhere | female | 30-34 | 0.623 | -5.2 (-5.8 to -4.1) stable |
| no → one child living elsewhere | female | 35-43 | 0.648 | -5.7 (-6.3 to -4.6) stable |

## Same person, one thing changed

The typical person: for numbers, the median of train-set people of that sex aged within a year of the stated age; for categories, the most common value. Then one feature is set to 'from' and then 'to'. Probabilities from the final model; change as above. 'The rest of this person' leaves out what the row changes.

| person | change | probability from → to | change, points | the rest of this person |
|---|---|---|---|---|
| male, 30 | earnings $20k → $60k | 0.688 → 0.797 | +10.9 (+9.9 to +11.8) stable | race other, some_college, protestant, south, 52 weeks, 180 cm, BMI 27.3, attendance 2, 0 children living elsewhere |
| female, 30 | earnings $20k → $60k | 0.760 → 0.776 | +1.6 (+1.8 to +3.2) stable | race other, some_college, protestant, south, 52 weeks, 163 cm, BMI 27.1, attendance 2, 0 children living elsewhere |
| male, 30 | no job → full-time all year at $40k | 0.547 → 0.763 | +21.6 (+20.1 to +24.5) stable | race other, some_college, protestant, south, 180 cm, BMI 27.3, attendance 2, 0 children living elsewhere |
| female, 30 | no job → full-time all year at $40k | 0.820 → 0.761 | -5.9 (-6.0 to -4.3) stable | race other, some_college, protestant, south, 163 cm, BMI 27.1, attendance 2, 0 children living elsewhere |
| male, 30 | high school → bachelor's | 0.748 → 0.736 | -1.2 (-3.6 to -0.2) stable | race other, protestant, south, earns $35,000, 52 weeks, 180 cm, BMI 27.3, attendance 2, 0 children living elsewhere |
| female, 30 | high school → bachelor's | 0.771 → 0.770 | -0.1 (-1.9 to +1.4) small | race other, protestant, south, earns $27,000, 52 weeks, 163 cm, BMI 27.1, attendance 2, 0 children living elsewhere |
| male, 30 | height 170 → 185 cm | 0.710 → 0.736 | +2.6 (+1.1 to +3.6) stable | race other, some_college, protestant, south, earns $35,000, 52 weeks, BMI 27.3, attendance 2, 0 children living elsewhere |
| female, 30 | BMI 22 → 32 | 0.748 → 0.768 | +2.0 (+0.8 to +3.8) stable | race other, some_college, protestant, south, earns $27,000, 52 weeks, 163 cm, attendance 2, 0 children living elsewhere |
| male, 28 | worship never → weekly | 0.688 → 0.743 | +5.4 (+4.0 to +6.7) stable | race other, some_college, protestant, south, earns $30,000, 52 weeks, 180 cm, BMI 27.0, 0 children living elsewhere |
| female, 28 | worship never → weekly | 0.741 → 0.760 | +1.9 (+1.1 to +3.4) stable | race other, some_college, protestant, south, earns $25,000, 51 weeks, 163 cm, BMI 26.5, 0 children living elsewhere |
| male, 30 | no → one child living elsewhere | 0.736 → 0.673 | -6.3 (-7.8 to -4.4) stable | race other, some_college, protestant, south, earns $35,000, 52 weeks, 180 cm, BMI 27.3, attendance 2 |
| female, 30 | no → one child living elsewhere | 0.763 → 0.698 | -6.5 (-6.8 to -4.6) stable | race other, some_college, protestant, south, earns $27,000, 52 weeks, 163 cm, BMI 27.1, attendance 2 |
