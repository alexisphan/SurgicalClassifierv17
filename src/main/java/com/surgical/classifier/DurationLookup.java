package com.surgical.classifier;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Loads real, measured anesthesia-duration data per operation name (opname),
 * derived from VitalDB's own anestart/aneend timestamps
 * (caseid_subjectid_casestart_caseend_.txt -> /data/opname_duration_lookup.csv).
 *
 * This replaces the rule-based Opioid/Duration estimator previously used in
 * TaxonomyClassifier.estimateDuration() with ground-truth average anesthesia
 * time (in minutes) bucketed into the same categories used throughout the
 * report (<1hr, <2hr, 1-2hr, 1-3hr, 2-4hr, 2-5hr, 3-6hr, >6hr).
 *
 * CSV format: opname,avg_duration_min,duration_bucket,n_samples
 */
public class DurationLookup {

    private final Map<String, String> bucketByOpname = new HashMap<>();
    private final Map<String, Double> avgMinutesByOpname = new HashMap<>();
    private final Map<String, Integer> sampleCountByOpname = new HashMap<>();

    public DurationLookup(String csvPath) {
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String header = br.readLine(); // skip header
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] parts = line.split(",", -1);
                if (parts.length < 4) continue;
                String opname = parts[0].trim();
                double avgMin;
                try {
                    avgMin = Double.parseDouble(parts[1].trim());
                } catch (NumberFormatException e) {
                    continue;
                }
                String bucket = parts[2].trim();
                int n;
                try {
                    n = Integer.parseInt(parts[3].trim());
                } catch (NumberFormatException e) {
                    n = 0;
                }
                bucketByOpname.put(opname.toLowerCase(), bucket);
                avgMinutesByOpname.put(opname.toLowerCase(), avgMin);
                sampleCountByOpname.put(opname.toLowerCase(), n);
            }
        } catch (IOException e) {
            System.err.println("WARNING: Could not load duration lookup from " + csvPath
                + " (" + e.getMessage() + "). Falling back to Unknown duration bucket.");
        }
    }

    /** Returns the measured duration bucket (e.g. "2-4hr") for a given opname, or "Unknown" if not found. */
    public String getBucket(String opname) {
        if (opname == null) return "Unknown";
        return bucketByOpname.getOrDefault(opname.trim().toLowerCase(), "Unknown");
    }

    /** Returns the measured average anesthesia duration in minutes, or -1 if not found. */
    public double getAvgMinutes(String opname) {
        if (opname == null) return -1;
        return avgMinutesByOpname.getOrDefault(opname.trim().toLowerCase(), -1.0);
    }

    /** Returns the number of VitalDB cases the average was computed from, or 0 if not found. */
    public int getSampleCount(String opname) {
        if (opname == null) return 0;
        return sampleCountByOpname.getOrDefault(opname.trim().toLowerCase(), 0);
    }

    public boolean hasData(String opname) {
        return opname != null && bucketByOpname.containsKey(opname.trim().toLowerCase());
    }

    public int size() {
        return bucketByOpname.size();
    }
}
