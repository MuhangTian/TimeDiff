DROP TABLE IF EXISTS patient_id;
CREATE TABLE patient_id AS

WITH patients_id AS (
	SELECT
		DISTINCT(patientunitstayid)
	FROM vitalPeriodic
),
disease_flag AS (
	SELECT patientunitstayid
	, MAX(CASE
		WHEN (icd9code LIKE '%995.92%') OR (icd9code LIKE '%995.91%') OR (icd9code LIKE '%785.52%') OR icd9code LIKE '%038%' THEN 1
		ELSE 0
		END) AS sepsis

	, MAX(CASE
		WHEN icd9code LIKE '%410%' THEN 1
		ELSE 0
		END) AS AMI

	, MAX(CASE
		WHEN icd9code LIKE '%428%' THEN 1
		ELSE 0
		END) AS heart_failure
	
	, MAX(CASE
		WHEN icd9code LIKE '%584%' THEN 1
		ELSE 0
		END) AS AKF

	FROM diagnosis
GROUP BY patientunitstayid
)
SELECT
	patients_id.patientunitstayid,

	CASE
		WHEN patient.hospitalDischargeStatus = 'Expired' THEN 1
		WHEN patient.hospitalDischargeLocation = 'Death' THEN 1
		WHEN patient.hospitalDischargeStatus IS NULL THEN NULL
		WHEN patient.hospitalDischargeLocation IS NULL THEN NULL 
		ELSE 0
	END AS hospital_expire_flag,

	disease_flag.sepsis::INTEGER,
	disease_flag.AMI::INTEGER,
	disease_flag.heart_failure::INTEGER,
	disease_flag.AKF::INTEGER
FROM patients_id
LEFT JOIN patient
	ON patients_id.patientunitstayid = patient.patientunitstayid
LEFT JOIN  disease_flag
	ON patients_id.patientunitstayid = disease_flag.patientunitstayid