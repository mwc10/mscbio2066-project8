SELECT md.chembl_id AS compound_chembl_id,
cs.canonical_smiles,
cs.standard_inchi,
cs.standard_inchi_key,
a.confidence_score,
act.standard_type,
act.standard_value,
act.standard_units,
act.pChEMBL_value,
td.chembl_id AS target_chembl_id,
td.organism,
td.pref_name,
cs.sequence,
pc.short_name,
pc.protein_class_desc
FROM target_dictionary td
  JOIN assays a ON td.tid = a.tid
  JOIN activities act ON a.assay_id = act.assay_id
  JOIN molecule_dictionary md ON act.molregno = md.molregno
  JOIN compound_structures cs ON md.molregno   = cs.molregno
  JOIN target_components   tc ON td.tid = tc.tid
  JOIN component_sequences cs ON tc.component_id = cs.component_id
  JOIN component_class    ccl ON tc.component_id = ccl.component_id
  JOIN protein_classification pc ON ccl.protein_class_id = pc.protein_class_id
    AND td.organism = 'Homo sapiens'
    AND a.confidence_score >= 6
    AND (act.standard_type = 'IC50' OR act.standard_type = 'EC50' OR act.standard_type = 'Ki' OR act.standard_type = 'Kd')
	AND act.standard_relation = '='
	AND pc.protein_class_desc LIKE '%kinase%';