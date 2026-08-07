/*
RETROSPECTIVE ADMISSION-LEVEL STUDY
===================================
Unit of analysis: one patient (subject_id), represented by the patient's latest
adult hospital admission. Admission time descending and hadm_id descending provide a
deterministic selection rule.

Covariate window: available vital signs and laboratory measurements recorded
within the first 120 hours after hospital admission. Hour 120 is not a
prediction landmark.

Retrospective targets:
1. hosp_mortality: any recorded in-hospital death during the same admission,
   whether death occurred before or after hour 120.
2. sepsis: any discharge diagnosis of severe sepsis (ICD-9 99592) or septic
   shock (ICD-9 78552) during the same admission.
3. septic_shock: any discharge diagnosis of septic shock (ICD-9 78552) during
   the same admission.

The ICD-9 data used here do not define onset time. Measurements may occur before
or after the clinical condition began. These are retrospective identification
tasks, not future-outcome predictions after a 120-hour landmark. Legacy column
names are retained for compatibility with existing analysis scripts.
*/
WITH params AS (
SELECT
120::int AS t_hours,
6::int AS vital_bin_hours,
24::int AS lab_bin_hours
),


base_adm AS (
SELECT
ad.subject_id,
ad.hadm_id,
ad.admittime,
p.gender,
CASE
WHEN EXTRACT(YEAR FROM age(ad.admittime, p.dob)) > 15 THEN 1
ELSE 0
END AS is_adult,
CASE
WHEN ad.deathtime IS NOT NULL THEN 1
ELSE 0
END AS hosp_mortality  -- retrospective same-admission in-hospital death
FROM mimic3.admissions ad
JOIN mimic3.patients p ON p.subject_id = ad.subject_id
),


ranked_adults AS (
    SELECT
        b.*,
        ROW_NUMBER() OVER (
            PARTITION BY b.subject_id
            ORDER BY b.admittime DESC, b.hadm_id DESC
        ) AS admission_recency_rank
    FROM base_adm b
    WHERE b.is_adult = 1
),

-- Prespecified one-admission-per-patient rule: retain the latest adult admission.
adults AS (
    SELECT
        subject_id,
        hadm_id,
        admittime,
        gender,
        is_adult,
        hosp_mortality
    FROM ranked_adults
    WHERE admission_recency_rank = 1
),

vital_bins AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.gender,
        a.hosp_mortality,
        gs.vbin
    FROM adults a
    CROSS JOIN LATERAL generate_series(
        0::bigint,
        (SELECT (t_hours / vital_bin_hours - 1)::bigint FROM params)
    ) AS gs(vbin)
),

lab_bins AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.gender,
        a.hosp_mortality,
        gs.lbin
    FROM adults a
    CROSS JOIN LATERAL generate_series(
        0::bigint,
        (SELECT (t_hours / lab_bin_hours - 1)::bigint FROM params)
    ) AS gs(lbin)
),

heart_rate_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (ce.charttime - a.admittime)) / 3600.0)
            / p.vital_bin_hours
        )::int AS vbin,
        AVG(ce.valuenum) AS value,
        'heart_rate' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.chartevents ce
      ON ce.subject_id = a.subject_id
     AND ce.hadm_id = a.hadm_id
    WHERE ce.itemid IN (211, 220045)
      AND ce.valuenum IS NOT NULL
      AND ce.charttime >= a.admittime
      AND ce.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, vbin
),

resp_rate_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (ce.charttime - a.admittime)) / 3600.0)
            / p.vital_bin_hours
        )::int AS vbin,
        AVG(ce.valuenum) AS value,
        'resp_rate' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.chartevents ce
      ON ce.subject_id = a.subject_id
     AND ce.hadm_id = a.hadm_id
    WHERE ce.itemid IN (219, 615, 618)
      AND ce.valuenum IS NOT NULL
      AND ce.charttime >= a.admittime
      AND ce.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, vbin
),

temperature_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (ce.charttime - a.admittime)) / 3600.0)
            / p.vital_bin_hours
        )::int AS vbin,
        AVG(ce.valuenum) AS value,
        'temperature' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.chartevents ce
      ON ce.subject_id = a.subject_id
     AND ce.hadm_id = a.hadm_id
    WHERE ce.itemid IN (676, 677, 223762)
      AND ce.valuenum IS NOT NULL
      AND ce.charttime >= a.admittime
      AND ce.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, vbin
),

sbp_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (ce.charttime - a.admittime)) / 3600.0)
            / p.vital_bin_hours
        )::int AS vbin,
        AVG(ce.valuenum) AS value,
        'sbp' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.chartevents ce
      ON ce.subject_id = a.subject_id
     AND ce.hadm_id = a.hadm_id
    WHERE ce.itemid IN (6, 51, 220179, 220050)
      AND ce.valuenum IS NOT NULL
      AND ce.charttime >= a.admittime
      AND ce.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, vbin
),

vitals_all AS (
    SELECT * FROM heart_rate_meas
    UNION ALL SELECT * FROM resp_rate_meas
    UNION ALL SELECT * FROM temperature_meas
    UNION ALL SELECT * FROM sbp_meas
),


wbc_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (le.charttime - a.admittime)) / 3600.0)
            / p.lab_bin_hours
        )::int AS lbin,
        AVG(le.valuenum) AS value,
        'wbc' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.labevents le
      ON le.hadm_id = a.hadm_id
    WHERE le.itemid IN (51300, 51301)
      AND le.valuenum IS NOT NULL
      AND le.charttime >= a.admittime
      AND le.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, lbin
),

platelet_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (le.charttime - a.admittime)) / 3600.0)
            / p.lab_bin_hours
        )::int AS lbin,
        AVG(le.valuenum) AS value,
        'platelets' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.labevents le
      ON le.hadm_id = a.hadm_id
    WHERE le.itemid = 51265
      AND le.valuenum IS NOT NULL
      AND le.charttime >= a.admittime
      AND le.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, lbin
),

creatinine_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (le.charttime - a.admittime)) / 3600.0)
            / p.lab_bin_hours
        )::int AS lbin,
        AVG(le.valuenum) AS value,
        'creatinine' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.labevents le
      ON le.hadm_id = a.hadm_id
    WHERE le.itemid = 50912
      AND le.valuenum IS NOT NULL
      AND le.charttime >= a.admittime
      AND le.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, lbin
),

inr_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (le.charttime - a.admittime)) / 3600.0)
            / p.lab_bin_hours
        )::int AS lbin,
        AVG(le.valuenum) AS value,
        'inr' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.labevents le
      ON le.hadm_id = a.hadm_id
    WHERE le.itemid = 51237
      AND le.valuenum IS NOT NULL
      AND le.charttime >= a.admittime
      AND le.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, lbin
),

lactate_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (le.charttime - a.admittime)) / 3600.0)
            / p.lab_bin_hours
        )::int AS lbin,
        AVG(le.valuenum) AS value,
        'lactate' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.labevents le
      ON le.hadm_id = a.hadm_id
    WHERE le.itemid = 50813
      AND le.valuenum IS NOT NULL
      AND le.charttime >= a.admittime
      AND le.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, lbin
),

bilirubin_meas AS (
    SELECT
        a.hadm_id,
        FLOOR(
            (EXTRACT(EPOCH FROM (le.charttime - a.admittime)) / 3600.0)
            / p.lab_bin_hours
        )::int AS lbin,
        AVG(le.valuenum) AS value,
        'bilirubin' AS feature_name
    FROM adults a
    CROSS JOIN params p
    JOIN mimic3.labevents le
      ON le.hadm_id = a.hadm_id
    WHERE le.itemid = 50885
      AND le.valuenum IS NOT NULL
      AND le.charttime >= a.admittime
      AND le.charttime <  a.admittime + p.t_hours * INTERVAL '1 hour'
    GROUP BY a.hadm_id, lbin
),

labs_all AS (
    SELECT * FROM wbc_meas
    UNION ALL SELECT * FROM platelet_meas
    UNION ALL SELECT * FROM creatinine_meas
    UNION ALL SELECT * FROM inr_meas
    UNION ALL SELECT * FROM lactate_meas
    UNION ALL SELECT * FROM bilirubin_meas
),

co_dx AS (
SELECT
hadm_id,
MAX(CASE WHEN icd9_code = '99592' THEN 1 ELSE 0 END) AS severe_sepsis,
MAX(CASE WHEN icd9_code = '78552' THEN 1 ELSE 0 END) AS septic_shock,
MAX(CASE WHEN icd9_code LIKE '250%' THEN 1 ELSE 0 END) AS diabetes,
MAX(CASE WHEN icd9_code BETWEEN '25040' AND '25093' THEN 1 ELSE 0 END) AS diabetes_complicated,
MAX(CASE WHEN icd9_code LIKE '585%' THEN 1 ELSE 0 END) AS ckd,
MAX(CASE WHEN icd9_code BETWEEN '5710' AND '5719' THEN 1 ELSE 0 END) AS chronic_liver_dz,
MAX(CASE WHEN icd9_code LIKE '428%' THEN 1 ELSE 0 END) AS chf,
MAX(CASE WHEN icd9_code BETWEEN '490' AND '496' THEN 1 ELSE 0 END) AS copd,
MAX(CASE WHEN icd9_code BETWEEN '491' AND '505' THEN 1 ELSE 0 END) AS chronic_pulmonary,
MAX(CASE WHEN icd9_code BETWEEN '07022' AND '07054' THEN 1 ELSE 0 END) AS liver,
MAX(CASE WHEN icd9_code BETWEEN '042' AND '0449' THEN 1 ELSE 0 END) AS aids,
MAX(CASE WHEN icd9_code BETWEEN '1960' AND '1991' THEN 1 ELSE 0 END) AS metastatic_cancer,
MAX(CASE WHEN icd9_code BETWEEN '1400' AND '1729' THEN 1 ELSE 0 END) AS solid_tumor,
MAX(CASE WHEN icd9_code BETWEEN '2910' AND '30503' THEN 1 ELSE 0 END) AS alcohol_abuse,
MAX(CASE WHEN icd9_code BETWEEN '30400' AND '30593' THEN 1 ELSE 0 END) AS drug_abuse,
MAX(CASE WHEN icd9_code BETWEEN '29500' AND '2989' THEN 1 ELSE 0 END) AS psychoses,
MAX(CASE WHEN icd9_code IN ('3004','30112','3090','3091','311') THEN 1 ELSE 0 END) AS depression,
MAX(CASE WHEN icd9_code BETWEEN '2760' AND '2769' THEN 1 ELSE 0 END) AS fluid_electrolyte
FROM mimic3.diagnoses_icd
GROUP BY hadm_id
),


per_vbin AS (
    SELECT
        vb.subject_id,
        vb.hadm_id,
        vb.admittime,
        vb.gender,
        vb.hosp_mortality,
        vb.vbin,
        MAX(CASE WHEN m.feature_name = 'heart_rate'  THEN m.value END) AS heart_rate,
        MAX(CASE WHEN m.feature_name = 'resp_rate'   THEN m.value END) AS resp_rate,
        MAX(CASE WHEN m.feature_name = 'temperature' THEN m.value END) AS temperature,
        MAX(CASE WHEN m.feature_name = 'sbp'         THEN m.value END) AS sbp
    FROM vital_bins vb
    LEFT JOIN vitals_all m
      ON m.hadm_id = vb.hadm_id
     AND m.vbin    = vb.vbin
    GROUP BY
        vb.subject_id, vb.hadm_id, vb.admittime,
        vb.gender, vb.hosp_mortality, vb.vbin
),

per_lbin AS (
    SELECT
        lb.subject_id,
        lb.hadm_id,
        lb.admittime,
        lb.gender,
        lb.hosp_mortality,
        lb.lbin,
        MAX(CASE WHEN m.feature_name = 'wbc'        THEN m.value END) AS wbc,
        MAX(CASE WHEN m.feature_name = 'platelets'  THEN m.value END) AS platelets,
        MAX(CASE WHEN m.feature_name = 'creatinine' THEN m.value END) AS creatinine,
        MAX(CASE WHEN m.feature_name = 'inr'        THEN m.value END) AS inr,
        MAX(CASE WHEN m.feature_name = 'lactate'    THEN m.value END) AS lactate,
        MAX(CASE WHEN m.feature_name = 'bilirubin'  THEN m.value END) AS bilirubin
    FROM lab_bins lb
    LEFT JOIN labs_all m
      ON m.hadm_id = lb.hadm_id
     AND m.lbin    = lb.lbin
    GROUP BY
        lb.subject_id, lb.hadm_id, lb.admittime,
        lb.gender, lb.hosp_mortality, lb.lbin
),

vital_pivot AS (
    SELECT
        subject_id,
        hadm_id,
        gender,
        admittime,
        hosp_mortality,
        array_agg(heart_rate  ORDER BY vbin) AS heart_rate_6h,
        array_agg(resp_rate   ORDER BY vbin) AS resp_rate_6h,
        array_agg(temperature ORDER BY vbin) AS temperature_6h,
        array_agg(sbp         ORDER BY vbin) AS sbp_6h
    FROM per_vbin
    GROUP BY subject_id, hadm_id, gender, admittime, hosp_mortality
),

lab_pivot AS (
    SELECT
        subject_id,
        hadm_id,
        array_agg(wbc        ORDER BY lbin) AS wbc_24h,
        array_agg(platelets  ORDER BY lbin) AS platelets_24h,
        array_agg(creatinine ORDER BY lbin) AS creatinine_24h,
        array_agg(inr        ORDER BY lbin) AS inr_24h,
        array_agg(lactate    ORDER BY lbin) AS lactate_24h,
        array_agg(bilirubin  ORDER BY lbin) AS bilirubin_24h
    FROM per_lbin
    GROUP BY subject_id, hadm_id
)

SELECT
v.subject_id,
v.hadm_id,
v.gender,
v.admittime,
v.hosp_mortality,
COALESCE(d.severe_sepsis,0) AS severe_sepsis,
COALESCE(d.septic_shock,0) AS septic_shock, -- retrospective ICD-9 78552
CASE WHEN COALESCE(d.severe_sepsis,0) = 1
OR COALESCE(d.septic_shock,0) = 1
THEN 1 ELSE 0 END AS sepsis, -- retrospective ICD-9 99592 or 78552


COALESCE(d.diabetes,0) AS diabetes,
COALESCE(d.diabetes_complicated,0) AS diabetes_complicated,
COALESCE(d.ckd,0) AS ckd,
COALESCE(d.chronic_liver_dz,0) AS chronic_liver_dz,
COALESCE(d.chf,0) AS chf,
COALESCE(d.copd,0) AS copd,
COALESCE(d.chronic_pulmonary,0) AS chronic_pulmonary,
COALESCE(d.liver,0) AS liver,
COALESCE(d.aids,0) AS aids,
COALESCE(d.metastatic_cancer,0) AS metastatic_cancer,
COALESCE(d.solid_tumor,0) AS solid_tumor,
COALESCE(d.alcohol_abuse,0) AS alcohol_abuse,
COALESCE(d.drug_abuse,0) AS drug_abuse,
COALESCE(d.psychoses,0) AS psychoses,
COALESCE(d.depression,0) AS depression,
COALESCE(d.fluid_electrolyte,0) AS fluid_electrolyte,


-- 20-bin (6h) vitals
v.heart_rate_6h,
v.resp_rate_6h,
v.temperature_6h,
v.sbp_6h,
-- 5-bin (24h) labs
l.wbc_24h,
l.platelets_24h,
l.creatinine_24h,
l.inr_24h,
l.lactate_24h,
l.bilirubin_24h
FROM vital_pivot v
LEFT JOIN lab_pivot l ON l.subject_id = v.subject_id AND l.hadm_id = v.hadm_id
LEFT JOIN co_dx d ON d.hadm_id = v.hadm_id
ORDER BY v.hadm_id;

-- SELECT
--     v.subject_id,
--     v.hadm_id,
--     v.gender,
--     v.admittime,
--     v.hosp_mortality,
--     COALESCE(d.severe_sepsis,0)       AS severe_sepsis,
--     COALESCE(d.septic_shock,0)        AS septic_shock,
--     CASE WHEN COALESCE(d.severe_sepsis,0) = 1
--            OR COALESCE(d.septic_shock,0) = 1
--          THEN 1 ELSE 0 END            AS sepsis,
--     COALESCE(d.diabetes,0)            AS diabetes,
--     COALESCE(d.ckd,0)                 AS ckd,
--     COALESCE(d.chronic_liver_dz,0)    AS chronic_liver_dz,
--     COALESCE(d.chf,0)                 AS chf,
--     COALESCE(d.copd,0)                AS copd,
--     -- 20-bin (6h) vitals
--     v.heart_rate_6h,
--     v.resp_rate_6h,
--     v.temperature_6h,
--     v.sbp_6h,
--     -- 5-bin (24h) labs
--     l.wbc_24h,
--     l.platelets_24h,
--     l.creatinine_24h,
--     l.inr_24h,
--     l.lactate_24h,
--     l.bilirubin_24h
-- FROM vital_pivot v
-- LEFT JOIN lab_pivot l ON l.subject_id = v.subject_id AND l.hadm_id = v.hadm_id
-- LEFT JOIN co_dx      d ON d.hadm_id = v.hadm_id
-- ORDER BY v.hadm_id;



