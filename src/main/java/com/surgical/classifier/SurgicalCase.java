package com.surgical.classifier;

/**
 * One row from vitaldb_surgery_classifiers.csv.
 * Columns: dx, opname, optype, approach, position, count
 *
 * After classification, holds all 6 taxonomy levels.
 */
public class SurgicalCase {

    // Input fields
    private final String dx, opname, optype, approach, position, count;

    // ML classification (used internally to derive taxonomy levels)
    private String mlCategory    = "";
    private double mlConfidence  = 0.0;
    private String mlMethod      = "";

    // 6 taxonomy levels
    private String l1Magnitude   = "";  // Major / Minor
    private String l2Domain      = "";  // Abdominal/GI, Thoracic, Neuro/Spine, etc.
    private String l3Approach    = "";  // Open, Laparoscopic, Robotic, etc.
    private String l4Subtype     = "";  // Colectomy, Craniotomy, Bariatric, etc.
    private String l5Modifiers   = "";  // Malignancy|Revision|Positioning
    private String l6Secondary   = "";  // Duration(measured)

    public SurgicalCase(String dx, String opname, String optype,
                        String approach, String position, String count) {
        this.dx       = safe(dx);
        this.opname   = safe(opname);
        this.optype   = safe(optype);
        this.approach = safe(approach);
        this.position = safe(position);
        this.count    = safe(count);
    }

    private static String safe(String s) { return s == null ? "" : s.trim(); }

    // ── Input getters ─────────────────────────────────────────────────────
    public String getDx()       { return dx; }
    public String getOpname()   { return opname; }
    public String getOptype()   { return optype; }
    public String getApproach() { return approach; }
    public String getPosition() { return position; }
    public String getCount()    { return count; }

    // ── ML result ─────────────────────────────────────────────────────────
    public String getMlCategory()             { return mlCategory; }
    public double getMlConfidence()           { return mlConfidence; }
    public String getMlMethod()               { return mlMethod; }
    public void setMlCategory(String s)       { this.mlCategory = s; }
    public void setMlConfidence(double d)     { this.mlConfidence = d; }
    public void setMlMethod(String s)         { this.mlMethod = s; }

    // Keep these aliases so WekaClassifier/FuzzyMatcher compile unchanged
    public String getPredictedCategory()          { return mlCategory; }
    public double getConfidenceScore()            { return mlConfidence; }
    public String getMatchMethod()                { return mlMethod; }
    public void setPredictedCategory(String s)    { this.mlCategory = s; }
    public void setConfidenceScore(double d)      { this.mlConfidence = d; }
    public void setMatchMethod(String s)          { this.mlMethod = s; }

    // ── Taxonomy level getters/setters ────────────────────────────────────
    public String getL1Magnitude()        { return l1Magnitude; }
    public String getL2Domain()           { return l2Domain; }
    public String getL3Approach()         { return l3Approach; }
    public String getL4Subtype()          { return l4Subtype; }
    public String getL5Modifiers()        { return l5Modifiers; }
    public String getL6Secondary()        { return l6Secondary; }

    public void setL1Magnitude(String s)  { this.l1Magnitude = s; }
    public void setL2Domain(String s)     { this.l2Domain = s; }
    public void setL3Approach(String s)   { this.l3Approach = s; }
    public void setL4Subtype(String s)    { this.l4Subtype = s; }
    public void setL5Modifiers(String s)  { this.l5Modifiers = s; }
    public void setL6Secondary(String s)  { this.l6Secondary = s; }

    /** Text blob for ML classification. */
    public String toTextBlob() {
        return ProcedureRecord.normalize(opname)   + " "
             + ProcedureRecord.normalize(dx)       + " "
             + ProcedureRecord.normalize(optype)   + " "
             + ProcedureRecord.normalize(approach);
    }
}
