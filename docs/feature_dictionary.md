# Official GBD Feature Dictionary & Input Specification

## 1. Country Identifier Requirements
Every input dataset must include a country identifier column matching one of the supported candidate names (`country`, `location`, `entity`, `iso3`, `iso_code`, `country_name`, `location_name`, `nation`):
- **Uniqueness:** Exactly one row per country or territory.
- **Completeness:** No null, blank, or whitespace identifiers.
- **Granularity:** Sovereign nations and recognized territories (GBD 204 locations). Regional and global aggregates are filtered prior to clustering.

---

## 2. Standard Metric Specification
All clustering indicators are measured as:
- **Measure:** DALYs (Disability-Adjusted Life Years), representing years of life lost due to premature mortality plus years lived with disability.
- **Metric:** Rate per 100,000 population.
- **Age Adjustment:** Age-standardized using the standard GBD global population age structure.
- **Sex:** Both sexes combined.

---

## 3. Selected Clustering Features (15 Mutually Exclusive Level 2 Causes)

### Non-Communicable Diseases (NCDs)

#### 1. `Cardiovascular diseases`
- **GBD Hierarchy:** Level 2 (Parent: Non-communicable diseases)
- **Clinical Definition:** Pathologies of the circulatory system, including ischemic heart disease, stroke, hypertensive heart disease, cardiomyopathy, myocarditis, endocarditis, and rheumatic heart disease.
- **Relevance:** The leading cause of premature adult mortality globally; elevated in aging populations, post-transition economies, and regions with high dietary sodium, tobacco use, and metabolic risks.
- **Measurement Caveats:** Vital registration completeness and stroke subtype differentiation vary across low-resource settings.

#### 2. `Neoplasms`
- **GBD Hierarchy:** Level 2 (Parent: Non-communicable diseases)
- **Clinical Definition:** All malignant neoplasms (cancers) spanning tracheal, bronchus, and lung, breast, colorectal, stomach, cervical, prostate, liver, and leukemia, among others.
- **Relevance:** Strongly associated with demographic aging, cumulative environmental and lifestyle exposures, and oncological healthcare capacity.
- **Measurement Caveats:** Underreporting occurs in regions lacking population-based cancer registries and biopsy diagnostic services.

#### 3. `Chronic respiratory diseases`
- **GBD Hierarchy:** Level 2 (Parent: Non-communicable diseases)
- **Clinical Definition:** Chronic obstructive pulmonary disease (COPD), asthma, pneumoconiosis, interstitial lung disease, and pulmonary sarcoidosis.
- **Relevance:** Captures chronic pulmonary health loss driven by ambient particulate matter, household air pollution (solid fuel burning), and smoking.
- **Measurement Caveats:** Diagnostic spirometry availability varies widely between high-income and low-income health systems.

#### 4. `Digestive diseases`
- **GBD Hierarchy:** Level 2 (Parent: Non-communicable diseases)
- **Clinical Definition:** Cirrhosis and other chronic liver diseases, peptic ulcer disease, gastritis, duodenitis, appendicitis, paralytic ileus, inflammatory bowel disease, and vascular intestinal disorders.
- **Relevance:** Differentiates countries with high burdens of viral hepatitis (HBV/HCV), alcohol use, and untreated acute surgical conditions.
- **Measurement Caveats:** Liver cirrhosis mortality attribution depends on diagnostic imaging and hospital documentation.

#### 5. `Neurological disorders`
- **GBD Hierarchy:** Level 2 (Parent: Non-communicable diseases)
- **Clinical Definition:** Alzheimer's disease and other dementias, Parkinson's disease, epilepsy, multiple sclerosis, motor neuron disease, headache disorders (migraine and tension-type), and other neurological conditions.
- **Relevance:** Disproportionately high disability and mortality burden in societies with high life expectancy and rapidly aging demographics.
- **Measurement Caveats:** Stigma, diagnostic criteria, and screening practices for early-stage dementias differ cross-nationally.

#### 6. `Mental disorders`
- **GBD Hierarchy:** Level 2 (Parent: Non-communicable diseases)
- **Clinical Definition:** Depressive disorders (major depressive disorder, dysthymia), anxiety disorders, bipolar disorder, schizophrenia, autism spectrum disorders, conduct disorder, attention-deficit/hyperactivity disorder, and eating disorders.
- **Relevance:** A dominant contributor to global Years Lived with Disability (YLDs), highlighting chronic non-fatal psychiatric burden.
- **Measurement Caveats:** Cultural reporting differences, community diagnostic survey methodologies, and psychiatric healthcare access impact measurement.

#### 7. `Substance use disorders`
- **GBD Hierarchy:** Level 2 (Parent: Non-communicable diseases)
- **Clinical Definition:** Alcohol use disorders and drug use disorders (opioids, cannabis, cocaine, amphetamines, and other illicit drugs).
- **Relevance:** Identifies distinct geographic and socioeconomic epidemics of substance dependence, acute overdose mortality, and chronic morbidity.
- **Measurement Caveats:** Legal prohibitions and cultural stigma create variable self-reporting rates in household health surveys.

#### 8. `Diabetes and kidney diseases`
- **GBD Hierarchy:** Level 2 (Parent: Non-communicable diseases)
- **Clinical Definition:** Type 1 and Type 2 diabetes mellitus, and chronic kidney disease (CKD) resulting from diabetes, hypertension, glomerulonephritis, or other causes.
- **Relevance:** Serves as a major tracer indicator for cardiometabolic transition, obesity prevalence, ultra-processed food consumption, and renal replacement access.
- **Measurement Caveats:** Fasting blood glucose measurement completeness and laboratory screening for renal impairment influence detection.

#### 9. `Musculoskeletal disorders`
- **GBD Hierarchy:** Level 2 (Parent: Non-communicable diseases)
- **Clinical Definition:** Low back pain, neck pain, osteoarthritis, rheumatoid arthritis, gout, and other musculoskeletal disorders.
- **Relevance:** A massive component of non-fatal morbidity across working-age and elderly populations; elevated in mechanized, sedentary, and service-based economies.
- **Measurement Caveats:** Primarily derived from epidemiological population health surveys rather than mortality registries.

---

### Communicable, Maternal, Neonatal, and Nutritional Diseases (CMNN)

#### 10. `Respiratory infections and tuberculosis`
- **GBD Hierarchy:** Level 2 (Parent: Communicable, maternal, neonatal, and nutritional diseases)
- **Clinical Definition:** Lower respiratory infections (pneumonia, bronchiolitis), upper respiratory infections, otitis media, and all forms of active tuberculosis (including drug-resistant strains).
- **Relevance:** A cornerstone indicator of child mortality, living conditions, overcrowding, and baseline public health infrastructure in low- and middle-income countries.
- **Measurement Caveats:** Clinical diagnosis vs. microbiological confirmation in sputum/blood cultures varies across rural settings.

#### 11. `Enteric infections`
- **GBD Hierarchy:** Level 2 (Parent: Communicable, maternal, neonatal, and nutritional diseases)
- **Clinical Definition:** Diarrheal diseases (caused by rotavirus, *E. coli*, *Shigella*, *Salmonella*, cholera, etc.), typhoid fever, paratyphoid fever, and other invasive enteric infections.
- **Relevance:** Strongly correlates with safe water, sanitation, and hygiene (WASH) infrastructure and pediatric malnutrition vulnerability.
- **Measurement Caveats:** Seasonality and episodic outbreak reporting introduce temporal fluctuations.

#### 12. `Neglected tropical diseases and malaria`
- **GBD Hierarchy:** Level 2 (Parent: Communicable, maternal, neonatal, and nutritional diseases)
- **Clinical Definition:** Malaria, lymphatic filariasis, onchocerciasis, schistosomiasis, soil-transmitted helminths, leishmaniasis, Chagas disease, African trypanosomiasis, cysticercosis, dengue, and rabies.
- **Relevance:** Highly specific geographic signature capturing tropical and subtropical endemic transmission environments, vector ecology, and structural poverty.
- **Measurement Caveats:** Rapid diagnostic testing completeness and vector control programs alter surveillance sensitivity.

#### 13. `Maternal and neonatal disorders`
- **GBD Hierarchy:** Level 2 (Parent: Communicable, maternal, neonatal, and nutritional diseases)
- **Clinical Definition:** Maternal complications (hemorrhage, sepsis, hypertensive disorders, obstructed labor), neonatal preterm birth complications, neonatal encephalopathy due to birth asphyxia, and neonatal sepsis/infections.
- **Relevance:** Essential tracer of obstetric care quality, skilled birth attendance, neonatal intensive care capacity, and primary health coverage.
- **Measurement Caveats:** Home births without birth certificates or maternal mortality audits require indirect modeling in demographic surveys.

#### 14. `Nutritional deficiencies`
- **GBD Hierarchy:** Level 2 (Parent: Communicable, maternal, neonatal, and nutritional diseases)
- **Clinical Definition:** Protein-energy malnutrition, dietary iron deficiency, vitamin A deficiency, iodine deficiency, and other micronutrient deficiencies.
- **Relevance:** Measures severe developmental vulnerability, food insecurity, stunting, wasting, and anemia, particularly among children and women of reproductive age.
- **Measurement Caveats:** Cutoffs for anthropometric surveys (e.g. WHO Child Growth Standards) and biological assays (serum ferritin) require standardization.

---

### Injuries

#### 15. `Self-harm and interpersonal violence`
- **GBD Hierarchy:** Level 2 (Parent: Injuries)
- **Clinical Definition:** Intentional self-harm (suicide), interpersonal violence (assault by firearm, sharp object, or other means), and collective violence/legal intervention.
- **Relevance:** Distinguishes countries experiencing acute social fragility, civil conflict, homicide epidemics, or elevated suicide mortality rates.
- **Measurement Caveats:** Suicide is subject to criminalization, social stigma, and misclassification (e.g., miscoded as accidental poisoning or drowning in vital statistics).
