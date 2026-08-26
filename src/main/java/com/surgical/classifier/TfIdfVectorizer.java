package com.surgical.classifier;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Lightweight TF-IDF vectorizer.
 * fit(corpus) → transform(doc) → cosineSimilarity(a, b)
 */
public class TfIdfVectorizer {

    private List<String>        vocabulary;
    private Map<String, Double> idf;
    private Map<String, Integer> termIndex;

    public void fit(List<String> documents) {
        int N = documents.size();
        Map<String, Integer> docFreq = new HashMap<>();
        for (String doc : documents) {
            Set<String> terms = new HashSet<>(tokenize(doc));
            for (String t : terms) docFreq.merge(t, 1, Integer::sum);
        }
        vocabulary = docFreq.entrySet().stream()
            .map(Map.Entry::getKey).sorted().collect(Collectors.toList());
        idf       = new HashMap<>();
        termIndex = new HashMap<>();
        for (int i = 0; i < vocabulary.size(); i++) {
            String t = vocabulary.get(i);
            termIndex.put(t, i);
            idf.put(t, Math.log((1.0 + N) / (1.0 + docFreq.getOrDefault(t, 0))) + 1.0);
        }
    }

    public double[] transform(String document) {
        List<String> tokens = tokenize(document);
        Map<String, Long> tf = tokens.stream()
            .collect(Collectors.groupingBy(t -> t, Collectors.counting()));
        double[] vec = new double[vocabulary.size()];
        double   len = tokens.isEmpty() ? 1 : tokens.size();
        for (Map.Entry<String, Long> e : tf.entrySet()) {
            Integer idx = termIndex.get(e.getKey());
            if (idx == null) continue;
            vec[idx] = (e.getValue() / len) * idf.getOrDefault(e.getKey(), 1.0);
        }
        return vec;
    }

    public static double cosineSimilarity(double[] a, double[] b) {
        double dot = 0, nA = 0, nB = 0;
        for (int i = 0; i < a.length; i++) { dot += a[i]*b[i]; nA += a[i]*a[i]; nB += b[i]*b[i]; }
        return (nA == 0 || nB == 0) ? 0 : dot / (Math.sqrt(nA) * Math.sqrt(nB));
    }

    public static List<String> tokenize(String text) {
        if (text == null || text.isBlank()) return Collections.emptyList();
        List<String> tokens = new ArrayList<>(
            Arrays.asList(ProcedureRecord.normalize(text).split("\\s+")));
        tokens.removeIf(t -> t.length() < 2 || STOP_WORDS.contains(t));
        return tokens;
    }

    public int vocabularySize() { return vocabulary == null ? 0 : vocabulary.size(); }

    private static final Set<String> STOP_WORDS = Set.of(
        "the","a","an","and","or","of","to","in","for","on","with","by","at",
        "from","as","is","it","be","was","are","were","this","that","using",
        "open","approach","procedure","surgery","surgical","operation"
    );
}
