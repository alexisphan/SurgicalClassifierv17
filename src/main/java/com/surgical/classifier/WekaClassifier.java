package com.surgical.classifier;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.*;

import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Trains a Weka Random Forest on TF-IDF vectors from the procedure reference file,
 * then classifies each SurgicalCase.
 */
public class WekaClassifier {

    private static final Logger LOG = Logger.getLogger(WekaClassifier.class.getName());

    private final TfIdfVectorizer vectorizer;
    private Classifier   model;
    private Instances    trainingHeader;
    private List<String> categoryLabels;

    public WekaClassifier(TfIdfVectorizer vectorizer) {
        this.vectorizer = vectorizer;
    }

    public void train(List<ProcedureRecord> procedures) throws Exception {
        if (procedures.isEmpty()) throw new IllegalArgumentException("No procedures to train on.");

        categoryLabels = procedures.stream()
            .map(ProcedureRecord::getProcCategory)
            .distinct().sorted()
            .collect(Collectors.toList());

        LOG.info("Training on " + procedures.size() + " records, "
                 + categoryLabels.size() + " categories, vocab=" + vectorizer.vocabularySize());

        Instances dataset = buildSchema("training", procedures.size());
        for (ProcedureRecord rec : procedures)
            dataset.add(buildInstance(rec.toTextBlob(), rec.getProcCategory(), dataset));

        RandomForest rf = new RandomForest();
        rf.setNumIterations(500);          // 500 trees — higher and more stable confidence
        rf.setNumExecutionSlots(Runtime.getRuntime().availableProcessors());
        rf.setSeed(42);
        rf.buildClassifier(dataset);

        this.model          = rf;
        this.trainingHeader = new Instances(dataset, 0);
        LOG.info("Training complete.");
    }

    public String classify(SurgicalCase sc) throws Exception {
        Instance inst = buildInstance(sc.toTextBlob(), null, trainingHeader);
        inst.setDataset(trainingHeader);
        double[] dist    = model.distributionForInstance(inst);
        int    bestIdx   = 0;
        double bestProb  = 0;
        for (int i = 0; i < dist.length; i++) {
            if (dist[i] > bestProb) { bestProb = dist[i]; bestIdx = i; }
        }
        String predicted = categoryLabels.get(bestIdx);
        sc.setPredictedCategory(predicted);
        sc.setConfidenceScore(bestProb);
        sc.setMatchMethod("RandomForest+TF-IDF");
        return predicted;
    }

    private Instances buildSchema(String name, int capacity) {
        ArrayList<Attribute> attrs = new ArrayList<>();
        for (int i = 0; i < vectorizer.vocabularySize(); i++) attrs.add(new Attribute("t" + i));
        attrs.add(new Attribute("category", new ArrayList<>(categoryLabels)));
        Instances data = new Instances(name, attrs, capacity);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    private Instance buildInstance(String text, String category, Instances schema) {
        double[] vec  = vectorizer.transform(text);
        double[] vals = new double[schema.numAttributes()];
        System.arraycopy(vec, 0, vals, 0, Math.min(vec.length, vals.length - 1));
        if (category != null) {
            int labelIdx = schema.classAttribute().indexOfValue(category);
            vals[schema.classIndex()] = (labelIdx >= 0) ? labelIdx : 0;
        }
        DenseInstance inst = new DenseInstance(1.0, vals);
        inst.setDataset(schema);
        return inst;
    }

    public List<String> getCategoryLabels() { return Collections.unmodifiableList(categoryLabels); }
}
