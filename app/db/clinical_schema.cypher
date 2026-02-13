// --- Reasoning Pipeline Graph Schema ---

// Step 4: Constraints for data integrity
CREATE CONSTRAINT encounter_id IF NOT EXISTS FOR (e:Encounter) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT state_name IF NOT EXISTS FOR (s:State) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT indicator_name IF NOT EXISTS FOR (i:Indicator) REQUIRE i.name IS UNIQUE;
CREATE CONSTRAINT intervention_name IF NOT EXISTS FOR (iv:Intervention) REQUIRE iv.name IS UNIQUE;

// Indices for performance
CREATE INDEX encounter_session IF NOT EXISTS FOR (e:Encounter) ON (e.session_id);
CREATE INDEX indicator_type IF NOT EXISTS FOR (i:Indicator) ON (i.type);

// --- Core Relationships ---
// Encounter -[RECORDED]-> Indicator
// Indicator -[INDICATES]-> State
// State -[REQUIRES_QUALITY]-> Property
// Property -[HAS_QUALITY]-(Intervention)
// Intervention -[PRODUCES]-> Outcome

// Example Bootstrap for Testing
MERGE (s:State {name: "Pitta↑", description: "Excess heat and acidity"})
MERGE (i:Indicator {name: "Burning Sensation", type: "Symptom"})
MERGE (i)-[:INDICATES {strength: 0.8}]->(s)
MERGE (p:Property {name: "Cooling", quality: "Sheeta"})
MERGE (s)-[:OPPOSED_BY]->(p)
MERGE (iv:Intervention {name: "Shatavari", type: "Herb"})
MERGE (iv)-[:HAS_QUALITY]->(p)
MERGE (iv)-[:PRODUCES]->(o:Outcome {name: "Acidity Relief"});
