package com.surgical.classifier;

import java.io.File;
import java.util.List;
import java.util.logging.*;
import java.util.stream.Collectors;

/**
 * ╔══════════════════════════════════════════════════════════════╗
 * ║   SURGICAL CASE CLASSIFIER  v6                              ║
 * ║   Anesthesia-Oriented 6-Level Hierarchical Taxonomy         ║
 * ║   Random Forest + TF-IDF  |  Fuzzy fallback                 ║
 * ╚══════════════════════════════════════════════════════════════╝
 *
 * IntelliJ Run Configuration:
 *   Main class        : com.surgical.classifier.Main
 *   Program arguments : data/procedures_reference.csv  data/cases_to_classify.csv  output
 *   Working directory : <folder containing pom.xml and data/>
 *
 * Output columns in classified_cases.csv:
 *   Original columns + ml_base_category + ml_confidence_pct +
 *   L1_magnitude | L2_domain | L3_approach | L4_subtype | L5_modifiers | L6_secondary_variables
 */
public class Main {

    private static final Logger LOG = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) throws Exception {
        configureLogging();
        printBanner();

        if (args.length < 2) {
            System.err.println("Usage: SurgicalClassifier <procedures.csv> <cases.csv> [output_dir]");
            System.exit(1);
        }
        String proceduresPath = args[0];
        String casesPath      = args[1];
        String outputDir      = args.length >= 3 ? args[2] : "output";
        new File(outputDir).mkdirs();

        // ── 1. Load ────────────────────────────────────────────────────────
        System.out.println("\n[1/5] Loading data...");
        List<ProcedureRecord> procedures = CsvLoader.loadProcedures(proceduresPath);
        List<SurgicalCase>    cases      = CsvLoader.loadCases(casesPath);
        System.out.printf("      Reference procedures : %,d%n", procedures.size());
        System.out.printf("      Cases to classify    : %,d%n", cases.size());

        if (procedures.isEmpty()) { System.err.println("ERROR: No procedures."); System.exit(1); }
        if (cases.isEmpty())      { System.err.println("ERROR: No cases.");      System.exit(1); }

        // ── 2. TF-IDF ──────────────────────────────────────────────────────
        System.out.println("\n[2/5] Building TF-IDF vocabulary...");
        TfIdfVectorizer vectorizer = new TfIdfVectorizer();
        vectorizer.fit(procedures.stream().map(ProcedureRecord::toTextBlob).collect(Collectors.toList()));
        System.out.printf("      Vocabulary size : %,d terms%n", vectorizer.vocabularySize());

        // ── 3. Train ───────────────────────────────────────────────────────
        System.out.println("\n[3/5] Training Random Forest (200 trees)...");
        long t0 = System.currentTimeMillis();
        WekaClassifier weka = new WekaClassifier(vectorizer);
        weka.train(procedures);
        System.out.printf("      Done in %.1f s  |  %d categories%n",
            (System.currentTimeMillis() - t0) / 1000.0, weka.getCategoryLabels().size());

        // ── 4. Classify — ML base + all 6 taxonomy levels ─────────────────
        System.out.println("\n[4/5] Classifying " + cases.size() + " cases across 6 taxonomy levels...");
        FuzzyMatcher     fuzzy    = new FuzzyMatcher(procedures);
        TaxonomyClassifier taxo   = new TaxonomyClassifier();

        // Load real, measured anesthesia-duration data (VitalDB anestart/aneend)
        // to replace the old rule-based Opioid/Duration guesses in Level 6.
        String durationLookupPath = new File(proceduresPath).getParent() == null
            ? "opname_duration_lookup.csv"
            : new File(new File(proceduresPath).getParent(), "opname_duration_lookup.csv").getPath();
        DurationLookup durationLookup = new DurationLookup(durationLookupPath);
        taxo.setDurationLookup(durationLookup);
        System.out.printf("      Measured duration lookup : %,d procedures (VitalDB anestart/aneend)%n",
            durationLookup.size());

        int n   = cases.size();
        int dot = Math.max(1, n / 40);
        int mlHits = 0, fuzzyHits = 0;

        for (int i = 0; i < n; i++) {
            SurgicalCase sc = cases.get(i);
            try {
                // Step A: ML base classification
                weka.classify(sc);
                mlHits++;
                if (sc.getMlConfidence() < FuzzyMatcher.ML_CONFIDENCE_THRESHOLD) {
                    fuzzy.applyFallback(sc);
                    fuzzyHits++;
                }

                // Step B: Derive all 6 taxonomy levels
                String l3 = taxo.classifyL3Approach(sc);
                String l1 = taxo.classifyL1Magnitude(sc);
                String l2 = taxo.classifyL2Domain(sc.getMlCategory(), sc);
                String l4 = taxo.classifyL4Subtype(sc.getMlCategory(), sc);
                String l5 = taxo.classifyL5Modifiers(sc);
                String l6 = taxo.classifyL6Secondary(sc, l1, l2, l4);

                sc.setL1Magnitude(l1);
                sc.setL2Domain(l2);
                sc.setL3Approach(l3);
                sc.setL4Subtype(l4);
                sc.setL5Modifiers(l5);
                sc.setL6Secondary(l6);

            } catch (Exception e) {
                LOG.warning("Failed on case " + i + ": " + e.getMessage());
                sc.setPredictedCategory("MiscellaneousOther");
                sc.setConfidenceScore(0.0);
                sc.setL1Magnitude("Major");
                sc.setL2Domain("Soft Tissue / Access / Misc");
                sc.setL3Approach(taxo.classifyL3Approach(sc));
                sc.setL4Subtype("Other Procedure");
                sc.setL5Modifiers("Unknown | Primary | Supine");
                sc.setL6Secondary("Est.Duration:2-4hr");
            }

            if ((i + 1) % dot == 0 || i == n - 1)
                System.out.printf("\r      Progress: %3d%%  (%,d / %,d)", (int)(100.0*(i+1)/n), i+1, n);
        }
        System.out.println();
        System.out.printf("      ML model (RF+TF-IDF) : %,d cases%n", mlHits);
        System.out.printf("      Fuzzy fallback        : %,d cases%n", fuzzyHits);

        // ── 5. Write outputs ───────────────────────────────────────────────
        System.out.println("\n[5/5] Writing outputs...");
        ReportWriter.writeCsv(cases,   outputDir + File.separator + "classified_cases.csv");
        ReportWriter.writeReport(cases, outputDir + File.separator + "classification_report.txt");

        System.out.println("\n══════════════════════════════════════════════════════");
        System.out.println("  Classification complete!");
        System.out.println("══════════════════════════════════════════════════════");
    }

    private static void configureLogging() {
        Logger root = Logger.getLogger("");
        root.setLevel(Level.WARNING);
        for (Handler h : root.getHandlers()) h.setLevel(Level.WARNING);
    }

    private static void printBanner() {
        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║   SURGICAL CASE CLASSIFIER  v6                      ║");
        System.out.println("║   Anesthesia-Oriented 6-Level Hierarchical Taxonomy  ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");
    }
}
