package com.surgical.classifier;

import org.apache.commons.text.similarity.JaroWinklerSimilarity;
import org.apache.commons.text.similarity.LevenshteinDistance;

import java.util.*;
import java.util.logging.Logger;

/**
 * Fuzzy string-matching fallback using Jaro-Winkler + Levenshtein.
 * Runs when ML confidence is below ML_CONFIDENCE_THRESHOLD.
 */
public class FuzzyMatcher {

    private static final Logger LOG = Logger.getLogger(FuzzyMatcher.class.getName());

    public static final double ML_CONFIDENCE_THRESHOLD = 0.55;

    private final List<ProcedureRecord> procedures;
    private final JaroWinklerSimilarity jaroWinkler = new JaroWinklerSimilarity();
    private final LevenshteinDistance   levenshtein  = new LevenshteinDistance(50);

    public FuzzyMatcher(List<ProcedureRecord> procedures) {
        this.procedures = procedures;
    }

    public void applyFallback(SurgicalCase sc) {
        String query   = ProcedureRecord.normalize(sc.getOpname() + " " + sc.getDx());
        double best    = -1;
        String bestCat = "MiscellaneousOther";

        for (ProcedureRecord rec : procedures) {
            double score = combined(query, ProcedureRecord.normalize(rec.getProcName()));
            if (score > best) { best = score; bestCat = rec.getProcCategory(); }
        }

        LOG.fine("Fuzzy [" + sc.getOpname() + "] → " + bestCat + " (" + String.format("%.3f", best) + ")");

        if (best > sc.getConfidenceScore()) {
            sc.setPredictedCategory(bestCat);
            sc.setConfidenceScore(best);
            sc.setMatchMethod("FuzzyMatch(Jaro-Winkler+Levenshtein)");
        } else {
            sc.setMatchMethod(sc.getMatchMethod() + "+FuzzyFallback");
        }
    }

    private double combined(String a, String b) {
        double jw  = jaroWinkler.apply(a, b);
        int maxLen = Math.max(a.length(), b.length());
        if (maxLen == 0) return 1.0;
        Integer ed  = levenshtein.apply(a, b);
        double lev  = (ed == null || ed < 0) ? 0.0 : 1.0 - (double) ed / maxLen;
        return 0.6 * jw + 0.4 * lev;
    }
}
