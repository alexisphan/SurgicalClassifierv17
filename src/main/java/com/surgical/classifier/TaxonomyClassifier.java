package com.surgical.classifier;

import org.apache.commons.text.similarity.JaroWinklerSimilarity;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Anesthesia-Oriented Hierarchical Surgical Taxonomy Classifier.
 *
 * Classifies each SurgicalCase across all 6 taxonomy levels:
 *
 *  L1 - Operative Magnitude   : Major / Minor
 *  L2 - Surgical Domain       : Abdominal/GI, Vascular, Neuro/Spine, etc.
 *  L3 - Surgical Approach     : Open, Laparoscopic, Robotic, Endoscopic, Percutaneous
 *  L4 - Procedure Subtype     : Colectomy, Bariatric, Craniotomy, etc.
 *  L5 - Modifiers             : Malignancy/Benign, Revision/Primary, Transplant, Positioning
 *  L6 - Secondary Variables   : Measured case duration (VitalDB)
 *
 * L3 and L5-positioning are derived directly from the approach/position columns.
 * L1, L2, L4 use ML (TF-IDF + Random Forest) via the WekaClassifier.
 * L5 modifiers and L6 are inferred from dx, opname, and optype fields.
 */
public class TaxonomyClassifier {

    private final JaroWinklerSimilarity jw = new JaroWinklerSimilarity();

    // ── Level 3: Approach ─────────────────────────────────────────────────
    // Mapped directly from the 'approach' column — already clean in the data.
    public String classifyL3Approach(SurgicalCase sc) {
        String a = sc.getApproach().toLowerCase().trim();
        if (a.contains("robotic"))                          return "Robotic";
        if (a.contains("videoscopic") || a.contains("laparoscopic") || a.contains("thoracoscopic"))
                                                            return "Laparoscopic";
        if (a.contains("endoscopic") || a.contains("bronchoscopic") || a.contains("transanal"))
                                                            return "Endoscopic";
        if (a.contains("percutaneous") || a.contains("interventional"))
                                                            return "Percutaneous";
        if (a.contains("open"))                             return "Open";
        return "Open"; // default — most general surgery is open if unspecified
    }

    // ── Level 1: Operative Magnitude ──────────────────────────────────────
    // Rules (in priority order):
    //   1. If opname explicitly matches a known major procedure → Major
    //   2. If opname explicitly matches a known minor procedure → Minor
    //   3. If optype is "Major resection" or "Transplantation" → Major
    //   4. Default → Major (this dataset skews heavily major)
    public String classifyL1Magnitude(SurgicalCase sc) {
        String op = sc.getOpname().toLowerCase().trim();
        String ot = sc.getOptype().toLowerCase().trim();

        // ── Explicit MAJOR list — checked first ──────────────────────────
        for (String kw : DEFINITE_MAJOR) {
            if (op.contains(kw)) return "Major";
        }

        // ── Explicit MINOR list — only if no major keyword matched ────────
        for (String kw : DEFINITE_MINOR) {
            if (op.equals(kw) || op.startsWith(kw + " ")) return "Minor";
        }
        for (String kw : DEFINITE_MINOR_CONTAINS) {
            if (op.contains(kw)) return "Minor";
        }

        // ── Optype fallback ───────────────────────────────────────────────
        if (ot.equals("major resection") || ot.equals("transplantation")) return "Major";

        return "Major"; // default — dataset skews major
    }

    // Procedures that are ALWAYS major regardless of optype label
    private static final List<String> DEFINITE_MAJOR = Arrays.asList(
        // Gastric
        "gastrectomy","esophagectomy","ivor lewis",
        // Colorectal
        "colectomy","anterior resection","abdominoperineal","proctocolectomy",
        "hartmann procedure","ileocolectomy","ileocecectomy",
        // Hepatobiliary/Pancreas
        "hepatectomy","pancreatectomy","pancreaticoduodenectomy","whipple",
        "hepatopancreaticoduodenectomy","choledochal cyst excision",
        "hepatic hilar resection","hepaticojejunostomy","liver transplant",
        "spleen preserving distal pancreatectomy","pancreatosplenectomy",
        // Thoracic
        "lobectomy","pneumonectomy","thymectomy","decortication","empyemectomy",
        "metastasectomy","lung segmentectomy","lung sgmentectomy",
        "resection of chest wall","resection of sternum","resection of rib",
        "thoracoplasty","thoracic sympathectomy","exploratory thoracotomy",
        "enucleation of esophageal tumor","closure of bronchopleural fistula",
        "clagett procedure","elosser operation","sternal fixation",
        "mediastinal lymph node dissection","sleeve lobectomy",
        // Vascular
        "bypass","aneurysmal repair","endarterectomy",
        "aortofemoral","aortoiliac","aortorenal",
        "resection of inferior vena cava","resection of lower limb artery",
        "resection of popliteal artery","vascular bypass","vessel graft",
        // Transplant
        "transplantation","donor hepatectomy","donor nephrectomy",
        // Gynecology/oncology
        "hysterectomy","debulking surgery","radical hysterectomy",
        "radical trachelectomy","radical parametrectomy","radical prostatectomy",
        "radical cystectomy","radical mastectomy",
        "ovarian cancer staging","pelvic tumorectomy",
        "paraaortic lymph node dissection","radical excision",
        "peritoneal mass excision","debulking",
        // Head and neck
        "neck dissection","laryngectomy","modified radical neck dissection",
        "selective neck dissection","radical excision of cervical lymph nodes",
        // Urology
        "nephrectomy","nephroureterectomy","radical prostatectomy",
        "partial cystectomy","radical cystectomy","pyeloplasty",
        // Ortho/reconstructive
        "resection of chest wall with reconstruction",
        "transplantation of gracilis muscle",
        // Other clearly major
        "adrenalectomy","splenectomy","exploratory laparotomy",
        "staging laparotomy","staging operation",
        "myomectomy","salpingo-oophorectomy","oophorectomy","salpingectomy",
        "orchiectomy","ovarian cystectomy",
        "axillary lymph node dissection","internal mammary lymph node dissection",
        "inguinal lymph node dissection","regional lymph node dissection",
        "pericardial window","fundoplication","adhesiolysis",
        "small bowel segmental resection","resection of intestine",
        "diverticulectomy","ampullectomy","transduodenal ampullectomy",
        "enucleation of pancreas","pancreatic cystojejunostomy",
        "total esophagectomy","esophagectomy","cervical esophagostomy",
        "cesarean section","pleurectomy"
    );

    // Procedures that are ALWAYS minor — exact match preferred
    private static final List<String> DEFINITE_MINOR = Arrays.asList(
        "biopsy","percutaneous biopsy","rigid bronchoscopy",
        "diagnostic laparoscopy","diagnostic pelviscopy","diagnostic procedure",
        "hemorrhoidectomy","anal fistulotomy","sphincterotomy","sphincteroplasty",
        "incision and drainage","incision of rectal stricture",
        "electrocauterization","macroplastique injection",
        "ligation and stripping","varicocelectomy",
        "sentinel lymph node mapping","thyroglossal duct cyst excision",
        "removal of hickman catheter","removal of staples","removal of foreign body",
        "removal of permanent catheter","removal of complicated graft",
        "insertion of permanent catheter","feeding jejunostomy",
        "cleaning of wound","closure of wound","exploration of wound",
        "debridement of wound","wound revision","bleeding control",
        "transurethral resection of bladder tumor","macdonald operation",
        "modified shirodkar operation","plication","pleurodesis",
        "rrectocele repair","rectal prolapse operation","delorme",
        "colostomy repair","ileostomy repair","marsupialization",
        "bullectomy","excision","wide excision","rigid bronchoscopic excision",
        "exploration","stoma repair"
    );

    // Minor — substring match (careful — these are unambiguous short procedures)
    private static final List<String> DEFINITE_MINOR_CONTAINS = Arrays.asList(
        "removal of","insertion of","reposition of","revision of arteriovenous",
        "closure of arteriovenous","transanal excision","transanal endoscopic",
        "hernia repair","inguinal hernia"
    );

    // ── Level 2: Surgical Domain ──────────────────────────────────────────
    // Mapped from ML predicted category (from WekaClassifier) — passed in as mlCategory.
    public String classifyL2Domain(String mlCategory, SurgicalCase sc) {
        switch (mlCategory) {
            case "Colorectal": case "Stomach": case "Biliary/Pancreas":
            case "Hepatic": case "Others":
                // Refine "Others" using opname/dx
                if (mlCategory.equals("Others")) return refineOthersDomain(sc);
                return "Abdominal / GI";
            case "Vascular":       return "Vascular";
            case "Major resection": case "Minor resection":
                // These two ML categories are VitalDB catch-alls that span multiple
                // anatomical domains (lung resections, but also nephrectomy, radical
                // prostatectomy, hysterectomy, salpingo-oophorectomy, ovarian
                // cystectomy, etc.) — they are NOT thoracic-specific. Refine using
                // opname/dx keywords instead of assuming Thoracic for all of them.
                return refineResectionDomain(sc);
            case "Breast": case "Thyroid":
                return "ENT / Head & Neck";
            case "Transplantation":
                return refineTransplantDomain(sc);
            // Neurosurgery categories
            case "AwakeCrani": case "BiopsyBrain": case "CraniEpilepsy":
            case "CraniNoS": case "CraniTumorNotSkullBase": case "CraniTumorSkullBase":
            case "CraniVessel": case "Cranioplasty": case "ETV":
            case "EVDShuntAndRevision": case "ElectrodeLeadPlacement":
            case "ElectrodeRemoval": case "EvacSubduralHematoma":
            case "IPGPlacementBatteryChange": case "MRIAblation":
            case "MVD": case "PituitaryTSA":
                return "Neuro / Spine";
            case "Spine": case "Rhizotomy":
                return "Neuro / Spine";
            case "Carotid":
                return "Vascular";
            case "PumpSCSPlacementRevision":
                return "Neuro / Spine";
            case "BiopsyMuscleNerve": case "CategoriesPG": case "MiscellaneousOther":
                return refineMiscDomain(sc);
            default:
                return refineMiscDomain(sc);
        }
    }

    // ── Level 4: Procedure Subtype ────────────────────────────────────────
    public String classifyL4Subtype(String mlCategory, SurgicalCase sc) {
        String op = sc.getOpname().toLowerCase();
        String ot = sc.getOptype().toLowerCase();

        // Colectomy / bowel resection
        if (op.contains("colectomy") || op.contains("anterior resection")
            || op.contains("abdominoperineal") || op.contains("proctocolectomy")
            || op.contains("hartmann") || op.contains("ileocolectomy")
            || op.contains("ileocecectomy") || op.contains("bowel")
            || op.contains("intestine") || op.contains("cecectomy")
            || ot.equals("colorectal"))
            return "Colectomy / Bowel Resection";

        // Bariatric
        if (op.contains("bariatric") || op.contains("gastric bypass")
            || op.contains("sleeve") || op.contains("gastric banding")
            || op.contains("duodenal switch"))
            return "Bariatric Surgery";

        // Craniotomy
        if (mlCategory.contains("Crani") || op.contains("craniotomy")
            || op.contains("craniectomy"))
            return "Craniotomy";

        // Spine
        if (mlCategory.equals("Spine") || mlCategory.equals("Rhizotomy")
            || op.contains("fusion") || op.contains("laminectomy")
            || op.contains("discectomy") || op.contains("decompression spine"))
            return "Spine Fusion / Decompression";

        // Cystoscopy / TURBT
        if (op.contains("cystoscopy") || op.contains("turbt")
            || op.contains("transurethral resection of bladder")
            || op.contains("cystectomy"))
            return "Cystoscopy / TURBT";

        // Thyroid / parathyroid
        if (ot.equals("thyroid") || op.contains("thyroid") || op.contains("parathyroid")
            || op.contains("thyroglossal") || op.contains("neck dissection"))
            return "Thyroid / Parathyroid";

        // Access / drainage
        if (op.contains("insertion of") || op.contains("removal of") || op.contains("drainage")
            || op.contains("incision and drainage") || op.contains("biopsy")
            || op.contains("catheter") || op.contains("feeding jejunostomy")
            || op.contains("colostomy") && !op.contains("resection")
            || op.contains("ileostomy") && !op.contains("resection"))
            return "Access / Drainage Procedures";

        // Gastric resection
        if (op.contains("gastrectomy") || ot.equals("stomach"))
            return "Gastric Resection";

        // Hepatobiliary
        if (op.contains("hepatectomy") || op.contains("liver") || op.contains("pancreatectomy")
            || op.contains("pancreaticoduodenectomy") || op.contains("whipple")
            || op.contains("cholecystectomy") || op.contains("bile duct")
            || ot.contains("biliary") || ot.equals("hepatic"))
            return "Hepatobiliary / Pancreatic Resection";

        // Thoracic resection
        if (op.contains("lobectomy") || op.contains("pneumonectomy")
            || op.contains("esophagectomy") || op.contains("lung")
            || op.contains("thymectomy") || op.contains("decortication"))
            return "Thoracic Resection";

        // Transplant
        if (ot.equals("transplantation") || op.contains("transplant"))
            return "Transplant / Major Reconstruction";

        // Vascular
        if (ot.equals("vascular") || op.contains("bypass") || op.contains("aneurysm")
            || op.contains("endarterectomy") || op.contains("fistula"))
            return "Vascular Reconstruction";

        // Breast
        if (ot.equals("breast") || op.contains("mastectomy") || op.contains("breast"))
            return "Breast Surgery";

        return "Other Procedure";
    }

    // ── Level 5: Modifiers ────────────────────────────────────────────────
    public String classifyL5Modifiers(SurgicalCase sc) {
        List<String> modifiers = new ArrayList<>();
        String op  = sc.getOpname().toLowerCase();
        String dx  = sc.getDx().toLowerCase();
        String pos = sc.getPosition().toLowerCase();

        // Malignancy vs benign
        if (dx.contains("cancer") || dx.contains("carcinoma") || dx.contains("malignant")
            || dx.contains("sarcoma") || dx.contains("lymphoma") || dx.contains("tumor")
            || dx.contains("neoplasm") || dx.contains("metastasis") || dx.contains("metastatic"))
            modifiers.add("Malignancy");
        else
            modifiers.add("Benign/Non-malignant");

        // Revision vs primary
        if (op.contains("revision") || op.contains("redo") || op.contains("completion")
            || op.contains("remnant") || op.contains("reoperative") || op.contains("repair"))
            modifiers.add("Revision");
        else
            modifiers.add("Primary");

        // Transplant / major reconstruction
        if (op.contains("transplant") || op.contains("reconstruction")
            || op.contains("free flap") || op.contains("bypass graft"))
            modifiers.add("Transplant/Reconstruction");

        // Positioning
        modifiers.add(classifyPositioning(pos));

        return String.join(" | ", modifiers);
    }

    public String classifyPositioning(String position) {
        String p = position.toLowerCase();
        if (p.contains("prone"))                   return "Prone";
        if (p.contains("lithotomy"))               return "Lithotomy";
        if (p.contains("lateral"))                 return "Lateral Decubitus";
        if (p.contains("trendelenburg"))           return "Trendelenburg";
        if (p.contains("reverse trendelenburg"))   return "Reverse Trendelenburg";
        if (p.contains("sitting"))                 return "Sitting";
        if (p.contains("kidney"))                  return "Lateral Kidney";
        if (p.contains("supine"))                  return "Supine";
        return "Supine";
    }

    // ── Level 6: Secondary Variables ─────────────────────────────────────
    // Anesthesia time under "Est.Duration" is now MEASURED, not estimated:
    // it is the average anestart→aneend interval for that exact opname,
    // computed directly from VitalDB's own case-level timestamps
    // (see DurationLookup / data/opname_duration_lookup.csv). This replaces
    // the rule-based Opioid/Duration guesses used in prior versions.
    private DurationLookup durationLookup;

    public void setDurationLookup(DurationLookup lookup) {
        this.durationLookup = lookup;
    }

    public String classifyL6Secondary(SurgicalCase sc, String l1Magnitude, String l2Domain, String l4Subtype) {
        List<String> vars = new ArrayList<>();
        String op  = sc.getOpname().toLowerCase();
        String dx  = sc.getDx().toLowerCase();
        String pos = sc.getPosition().toLowerCase();

        // Measured anesthesia duration (VitalDB anestart/aneend ground truth)
        vars.add("Est.Duration:" + lookupMeasuredDuration(sc, l1Magnitude, l2Domain, l4Subtype, op));

        return String.join(" | ", vars);
    }

    // ── L6 sub-estimators ─────────────────────────────────────────────────

    /**
     * Returns the real, measured average anesthesia-duration bucket for this case's
     * opname, sourced from VitalDB's own anestart/aneend timestamps. Falls back to
     * the prior rule-based estimate only if the opname has no measured data
     * (e.g. a brand-new procedure name not present in VitalDB's case database).
     */
    private String lookupMeasuredDuration(SurgicalCase sc, String mag, String domain, String subtype, String op) {
        if (durationLookup != null && durationLookup.hasData(sc.getOpname())) {
            return durationLookup.getBucket(sc.getOpname());
        }
        return ruleBasedDurationFallback(mag, subtype);
    }

    /** Fallback only — used when an opname has no VitalDB-measured duration available. */
    private String ruleBasedDurationFallback(String mag, String subtype) {
        if (subtype.contains("Transplant"))           return ">6hr";
        if (subtype.contains("Hepatobiliary"))        return "3-6hr";
        if (subtype.contains("Thoracic"))             return "3-6hr";
        if (subtype.contains("Craniotomy"))           return "3-6hr";
        if (subtype.contains("Spine Fusion"))         return "2-5hr";
        if (subtype.contains("Colectomy"))            return "2-4hr";
        if (subtype.contains("Gastric"))              return "2-4hr";
        if (subtype.contains("Vascular"))             return "2-5hr";
        if (subtype.contains("Bariatric"))            return "1-3hr";
        if (subtype.contains("Breast"))               return "1-3hr";
        if (subtype.contains("Thyroid"))              return "1-2hr";
        if (subtype.contains("Access"))               return "<1hr";
        if (subtype.contains("Cystoscopy"))           return "<1hr";
        if (mag.equals("Minor"))                      return "<2hr";
        return "2-4hr";
    }

    // ── Domain refinement helpers ─────────────────────────────────────────

    private String refineOthersDomain(SurgicalCase sc) {
        String op = sc.getOpname().toLowerCase();
        String dx = sc.getDx().toLowerCase();
        if (op.contains("hysterectomy") || op.contains("oophorectomy")
            || op.contains("salpingectomy") || op.contains("myomectomy")
            || op.contains("ovarian") || op.contains("cesarean")
            || dx.contains("uterine") || dx.contains("endometrial")
            || dx.contains("ovarian") || dx.contains("cervical cancer")
            || dx.contains("myoma"))
            return "Abdominal / GI"; // GYN treated as abdominal for anesthesia purposes
        if (op.contains("nephrectomy") || op.contains("prostatectomy")
            || op.contains("cystectomy") || op.contains("ureter")
            || op.contains("pyeloplasty") || op.contains("orchiectomy")
            || op.contains("transurethral") || op.contains("bladder"))
            return "Urology";
        if (op.contains("hernia"))
            return "Abdominal / GI";
        if (op.contains("adrenalectomy"))
            return "Abdominal / GI";
        if (op.contains("splenectomy"))
            return "Abdominal / GI";
        if (op.contains("thyroid") || op.contains("parathyroid")
            || op.contains("neck dissection") || op.contains("thyroglossal"))
            return "ENT / Head & Neck";
        return "Soft Tissue / Access / Misc";
    }

    /**
     * Disambiguates the "Major resection" / "Minor resection" ML catch-all
     * categories, which mix lung procedures with GYN and urologic procedures
     * in the reference taxonomy (procedures_reference.csv).
     */
    private String refineResectionDomain(SurgicalCase sc) {
        String op = sc.getOpname().toLowerCase();
        String dx = sc.getDx().toLowerCase();

        // Thoracic / lung
        if (op.contains("lung") || op.contains("lobectomy") || op.contains("pneumonectomy")
            || op.contains("bullectomy") || op.contains("segmentectomy")
            || op.contains("sgmentectomy") || op.contains("wedge resection")
            || op.contains("thoracic") || op.contains("thoracotomy")
            || op.contains("metastasectomy") || op.contains("decortication")
            || op.contains("thymectomy") || op.contains("pleurectomy"))
            return "Thoracic";

        // Gynecologic — treated as abdominal for anesthesia purposes, matching
        // the convention already used in refineOthersDomain().
        if (op.contains("hysterectomy") || op.contains("oophorectomy")
            || op.contains("salpingectomy") || op.contains("myomectomy")
            || op.contains("ovarian") || op.contains("cesarean")
            || dx.contains("uterine") || dx.contains("endometrial")
            || dx.contains("ovarian") || dx.contains("cervical cancer")
            || dx.contains("myoma"))
            return "Abdominal / GI";

        // Urology
        if (op.contains("nephrectomy") || op.contains("prostatectomy")
            || op.contains("cystectomy") || op.contains("ureter")
            || op.contains("pyeloplasty") || op.contains("orchiectomy")
            || op.contains("transurethral") || op.contains("bladder"))
            return "Urology";

        // Default: this bucket's largest remaining share is lung procedures.
        return "Thoracic";
    }

    private String refineTransplantDomain(SurgicalCase sc) {
        String op = sc.getOpname().toLowerCase();
        if (op.contains("kidney") || op.contains("renal") || op.contains("nephrectomy"))
            return "Urology";
        if (op.contains("liver") || op.contains("hepat"))
            return "Abdominal / GI";
        if (op.contains("lung"))
            return "Thoracic";
        if (op.contains("pancreas"))
            return "Abdominal / GI";
        return "Abdominal / GI";
    }

    private String refineMiscDomain(SurgicalCase sc) {
        String op = sc.getOpname().toLowerCase();
        if (op.contains("biopsy") || op.contains("excision") || op.contains("debridement")
            || op.contains("wound") || op.contains("incision") || op.contains("drain")
            || op.contains("catheter") || op.contains("removal"))
            return "Soft Tissue / Access / Misc";
        if (op.contains("breast") || op.contains("mastectomy"))
            return "ENT / Head & Neck";
        return "Soft Tissue / Access / Misc";
    }

}
