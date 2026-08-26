package com.surgical.classifier;

import com.opencsv.CSVWriter;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

public class ReportWriter {

    private static final DateTimeFormatter FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    // ── CSV ───────────────────────────────────────────────────────────────
    public static void writeCsv(List<SurgicalCase> cases, String path) throws IOException {
        try (CSVWriter w = new CSVWriter(
                new OutputStreamWriter(new FileOutputStream(path), StandardCharsets.UTF_8))) {

            // Header — original columns + ML base + all 6 taxonomy levels
            w.writeNext(new String[]{
                "dx","opname","optype","approach","position","count",
                "ml_base_category","ml_confidence_pct",
                "Level1_magnitude","Level2_domain","Level3_approach",
                "Level4_subtype","Level5_modifiers","Level6_secondary_variables"
            });

            for (SurgicalCase sc : cases) {
                w.writeNext(new String[]{
                    sc.getDx(), sc.getOpname(), sc.getOptype(),
                    sc.getApproach(), sc.getPosition(), sc.getCount(),
                    sc.getMlCategory(),
                    String.format("%.1f", sc.getMlConfidence() * 100),
                    sc.getL1Magnitude(),
                    sc.getL2Domain(),
                    sc.getL3Approach(),
                    sc.getL4Subtype(),
                    sc.getL5Modifiers(),
                    sc.getL6Secondary()
                });
            }
        }
        System.out.println("  CSV     → " + new File(path).getAbsolutePath());
    }

    // ── Text report ───────────────────────────────────────────────────────
    public static void writeReport(List<SurgicalCase> cases, String path) throws IOException {
        int total    = cases.size();
        int lowConf  = (int) cases.stream().filter(c -> c.getMlConfidence() < 0.50).count();
        double avg   = cases.stream().mapToDouble(SurgicalCase::getMlConfidence).average().orElse(0);

        try (PrintWriter pw = new PrintWriter(
                new OutputStreamWriter(new FileOutputStream(path), StandardCharsets.UTF_8))) {

            line(pw, "=", 80);
            pw.println("  ANESTHESIA-ORIENTED SURGICAL TAXONOMY — CLASSIFICATION REPORT");
            pw.println("  Generated : " + LocalDateTime.now().format(FMT));
            line(pw, "=", 80);
            pw.println();

            // Summary
            pw.println("SUMMARY");
            line(pw, "-", 40);
            pw.printf("  Total cases classified      : %,d%n", total);
            pw.printf("  Average ML confidence       : %.1f%%%n", avg * 100);
            pw.printf("  Low confidence (<50%%)       : %,d  ← review recommended%n", lowConf);
            pw.println();

            // L1 breakdown
            printBreakdown(pw, cases, "LEVEL 1 — OPERATIVE MAGNITUDE",
                cases.stream().collect(Collectors.groupingBy(SurgicalCase::getL1Magnitude, Collectors.counting())));

            // L2 breakdown
            printBreakdown(pw, cases, "LEVEL 2 — SURGICAL DOMAIN",
                cases.stream().collect(Collectors.groupingBy(SurgicalCase::getL2Domain, Collectors.counting())));

            // L3 breakdown
            printBreakdown(pw, cases, "LEVEL 3 — SURGICAL APPROACH",
                cases.stream().collect(Collectors.groupingBy(SurgicalCase::getL3Approach, Collectors.counting())));

            // L4 breakdown
            printBreakdown(pw, cases, "LEVEL 4 — PROCEDURE SUBTYPE",
                cases.stream().collect(Collectors.groupingBy(SurgicalCase::getL4Subtype, Collectors.counting())));

            // L5 positioning — always the LAST pipe-separated element
            Map<String, Long> posFreq = cases.stream()
                .map(c -> extractLastModifier(c.getL5Modifiers()))
                .collect(Collectors.groupingBy(s -> s, Collectors.counting()));
            printBreakdown(pw, cases, "LEVEL 5 — POSITIONING (from modifiers)", posFreq);

            // L5 malignancy
            Map<String, Long> malFreq = cases.stream()
                .map(c -> extractModifier(c.getL5Modifiers(), 0))
                .collect(Collectors.groupingBy(s -> s, Collectors.counting()));
            printBreakdown(pw, cases, "LEVEL 5 — MALIGNANCY STATUS", malFreq);

            // L6 estimated duration
            Map<String, Long> durFreq = cases.stream()
                .map(c -> extractL6Field(c.getL6Secondary(), "Est.Duration"))
                .collect(Collectors.groupingBy(s -> s, Collectors.counting()));
            printBreakdown(pw, cases, "LEVEL 6 — ESTIMATED CASE DURATION", durFreq);

            // Low confidence flagging
            List<SurgicalCase> flagged = cases.stream()
                .filter(c -> c.getMlConfidence() < 0.50)
                .sorted(Comparator.comparingDouble(SurgicalCase::getMlConfidence))
                .collect(Collectors.toList());
            if (!flagged.isEmpty()) {
                pw.println("LOW-CONFIDENCE CASES — MANUAL REVIEW RECOMMENDED");
                line(pw, "-", 80);
                pw.printf("  %-35s  %-22s  %-18s  %s%n","OPNAME","LEVEL 2 DOMAIN","LEVEL 4 SUBTYPE","CONF%");
                line(pw, "-", 80);
                for (SurgicalCase sc : flagged) {
                    pw.printf("  %-35s  %-22s  %-18s  %.1f%%%n",
                        trunc(sc.getOpname(),35), trunc(sc.getL2Domain(),22),
                        trunc(sc.getL4Subtype(),18), sc.getMlConfidence()*100);
                }
                pw.println();
            }

            line(pw, "=", 80);
            pw.println("  END OF REPORT");
            line(pw, "=", 80);
        }
        System.out.println("  Report  → " + new File(path).getAbsolutePath());
    }

    // ── Helpers ───────────────────────────────────────────────────────────
    private static void printBreakdown(PrintWriter pw, List<SurgicalCase> cases,
                                       String title, Map<String, Long> freq) {
        int total = cases.size();
        pw.println(title);
        line(pw, "-", 50);
        freq.entrySet().stream()
            .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
            .forEach(e -> pw.printf("  %-45s  %5d  (%5.1f%%)%n",
                e.getKey(), e.getValue(), 100.0 * e.getValue() / total));
        pw.println();
    }

    /** Extracts a pipe-delimited modifier by index (0=malignancy,1=revision,...) */
    private static String extractModifier(String modifiers, int idx) {
        if (modifiers == null || modifiers.isBlank()) return "Unknown";
        String[] parts = modifiers.split("\\|");
        return (idx < parts.length) ? parts[idx].trim() : "Unknown";
    }

    /** Extracts the LAST pipe-delimited modifier — positioning is always last. */
    private static String extractLastModifier(String modifiers) {
        if (modifiers == null || modifiers.isBlank()) return "Unknown";
        String[] parts = modifiers.split("\\|");
        return parts[parts.length - 1].trim();
    }

    /** Extracts a key:value field from L6 secondary string */
    private static String extractL6Field(String l6, String key) {
        if (l6 == null) return "Unknown";
        for (String part : l6.split("\\|")) {
            String p = part.trim();
            if (p.startsWith(key + ":")) return p.substring(key.length() + 1).trim();
        }
        return "Unknown";
    }

    private static void line(PrintWriter pw, String ch, int n) { pw.println(ch.repeat(n)); }
    private static String trunc(String s, int n) {
        if (s == null) return "";
        return s.length() <= n ? s : s.substring(0, n-1) + "…";
    }
}
