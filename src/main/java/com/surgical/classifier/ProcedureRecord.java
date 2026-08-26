package com.surgical.classifier;

/**
 * One row from the reference file: proc_id, proc_name, proc_category.
 * The reference file is Neurosurgical_Cases_Categories.xlsx (converted to CSV).
 */
public class ProcedureRecord {

    private final String procId;
    private final String procName;
    private final String procCategory;

    public ProcedureRecord(String procId, String procName, String procCategory) {
        this.procId       = safe(procId);
        this.procName     = safe(procName);
        this.procCategory = safe(procCategory);
    }

    private static String safe(String s) { return s == null ? "" : s.trim(); }

    public String getProcId()       { return procId; }
    public String getProcName()     { return procName; }
    public String getProcCategory() { return procCategory; }

    /**
     * Text blob used for TF-IDF training.
     * Procedure name carries all the signal — category is the label, not a feature.
     */
    public String toTextBlob() {
        return normalize(procName);
    }

    public static String normalize(String s) {
        if (s == null) return "";
        return s.toLowerCase()
                .replaceAll("[^a-z0-9\\s]", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }
}
