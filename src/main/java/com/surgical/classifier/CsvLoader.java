package com.surgical.classifier;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.logging.Logger;

public class CsvLoader {

    private static final Logger LOG = Logger.getLogger(CsvLoader.class.getName());

    // ── Reference file: proc_id, proc_name, proc_category ─────────────────
    public static List<ProcedureRecord> loadProcedures(String filePath) throws IOException {
        List<ProcedureRecord> records = new ArrayList<>();
        try (CSVReader reader = openReader(filePath)) {
            String[] headers = reader.readNext();
            if (headers == null) throw new IOException("Empty file: " + filePath);
            Map<String, Integer> idx = index(headers);
            int idCol   = require(idx, filePath, "proc_id");
            int nameCol = require(idx, filePath, "proc_name");
            int catCol  = require(idx, filePath, "proc_category");
            String[] row;
            while ((row = reader.readNext()) != null) {
                if (allBlank(row)) continue;
                String cat = get(row, catCol);
                if (cat.isEmpty()) continue; // skip rows without a category
                records.add(new ProcedureRecord(get(row, idCol), get(row, nameCol), cat));
            }
        } catch (CsvValidationException e) {
            throw new IOException("CSV parse error: " + filePath, e);
        }
        LOG.info("Loaded " + records.size() + " procedure records from " + filePath);
        return records;
    }

    // ── Cases file: dx, opname, optype, approach, position, count ─────────
    public static List<SurgicalCase> loadCases(String filePath) throws IOException {
        List<SurgicalCase> cases = new ArrayList<>();
        try (CSVReader reader = openReader(filePath)) {
            String[] headers = reader.readNext();
            if (headers == null) throw new IOException("Empty file: " + filePath);
            Map<String, Integer> idx = index(headers);
            int dxCol       = require(idx, filePath, "dx");
            int opnameCol   = require(idx, filePath, "opname");
            int optypeCol   = require(idx, filePath, "optype");
            int approachCol = require(idx, filePath, "approach");
            int positionCol = require(idx, filePath, "position");
            int countCol    = require(idx, filePath, "count");
            String[] row;
            while ((row = reader.readNext()) != null) {
                if (allBlank(row)) continue;
                cases.add(new SurgicalCase(
                    get(row, dxCol), get(row, opnameCol), get(row, optypeCol),
                    get(row, approachCol), get(row, positionCol), get(row, countCol)
                ));
            }
        } catch (CsvValidationException e) {
            throw new IOException("CSV parse error: " + filePath, e);
        }
        LOG.info("Loaded " + cases.size() + " cases from " + filePath);
        return cases;
    }

    // ── Helpers ───────────────────────────────────────────────────────────
    private static CSVReader openReader(String path) throws IOException {
        return new CSVReader(new InputStreamReader(new FileInputStream(path), StandardCharsets.UTF_8));
    }
    private static Map<String, Integer> index(String[] headers) {
        Map<String, Integer> m = new LinkedHashMap<>();
        for (int i = 0; i < headers.length; i++) m.put(headers[i].trim().toLowerCase(), i);
        return m;
    }
    private static int require(Map<String, Integer> idx, String file, String col) throws IOException {
        Integer i = idx.get(col.toLowerCase());
        if (i == null) throw new IOException(
            "Column '" + col + "' not found in " + file + ". Found: " + idx.keySet());
        return i;
    }
    private static String get(String[] row, int col) {
        return (col < row.length && row[col] != null) ? row[col].trim() : "";
    }
    private static boolean allBlank(String[] row) {
        for (String c : row) if (c != null && !c.isBlank()) return false;
        return true;
    }
}
