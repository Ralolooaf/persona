# Persona audit (run PA2): pre-registration

Code: `persona_audit.py` (persona_audit/2.0). Kaggle cell hash `b5d1d2c4f506615f` (first 16 hex digits of SHA-256 over the file without trailing newlines; the run prints it and stores it in its registration). File SHA-256 `b36fac8b71ee386a9a4457cc61f8abd0fd74091797334556ee7c3ce7db229677`.

**Predictions hash** `db4982056790f226`: the first 16 hex digits of SHA-256 over `json.dumps({'predictions': PREDICTIONS, 'primary': PRIMARY, 'alpha': ALPHA}, sort_keys=True)`. The run prints it, stores it in `prereg_pa2_PA2.json` and its report checks it.

## Design

- **model**: Qwen/Qwen3-8B
- **conditions**: D = default assistant, no card, thinking off (vLLM); P = character card, thinking off (vLLM); T = character card, thinking on (vLLM); R = character card, thinking on, told to think as the character (vLLM); S0 = character card, thinking off, transformers generate, no amplification (the control); S_trait = S0 plus alpha x the character's twin-contrast trait direction; S_role = S0 plus alpha x the character's whole card shift (card minus default, as in role steering); S_axis = S0 plus alpha x the mean card shift of the 16 characters (the assistant-axis direction on these cards)
- **samples**: {"D": {"dil": 4, "self": 4, "calib": 4}, "P": {"dil": 2, "self": 2, "calib": 2}, "T": {"dil": 2}, "R": {"dil": 2}, "S0": {"dil": 2}, "S_trait": {"dil": 2}, "S_role": {"dil": 2}, "S_axis": {"dil": 2}}
- **sampling**: {"temperature": 0.7, "top_p": 0.95, "top_k": "off"}
- **max_tokens**: {"answer": 160, "thinking": 700, "after_forcing": 160}
- **budget_forcing**: a thinking sample that has not closed its thinking after 700 tokens gets 'I have to decide now.' and the end of thinking, then up to 160 answer tokens
- **fallback**: an answer without a CHOICE / ANSWER line is completed greedily (3 tokens) after the line's label
- **feature_layers**: [12, 20, 28]
- **steer_layer**: 20
- **decision_point**: the hidden state (output of each feature layer) at the last prompt token of each dilemma prompt, after the chat template's empty thinking block and before any answer token; D and P
- **decomposition**: for each dilemma, the 16 characters' decision-point shifts from D are split by the 2 x 2^3 factorial with +-1 codes: shared = the mean shift; style, h, r, g = the main effects; the 11 interactions (with which the names are confounded); squared norms, so mean squared shift = shared + sum of the 15 contrasts, and the between-character part = the sum of the contrasts
- **directions**: from the mean hidden state (output of layer 20, mean over answer tokens) of the P answers to the 8 calibration questions: m_c per character and m_D for the default; role: v_c = m_c - m_D; axis: the mean of v_c over the 16 characters; trait: the sum over h, r, g of (+1 for the character's counter-default pole, -1 for its default pole) x (mean m of the 8 characters with the counter pole - mean m of the 8 with the default pole), which is the mean difference over the 8 twin pairs; trait and axis are scaled to |v_c| for each character, so all three have the same size
- **steering**: S adds alpha x the direction to the output of layer 20 at every position, prompt and answer
- **alpha_grid**: [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
- **alpha_rule**: for each direction separately: the 16 characters answer the first 4 calibration questions and 4 neutral preference choices once per alpha, steered; alpha is acceptable if the distinct-2 of the calibration answers >= 0.6 x that of P's answers to the same 4 questions (first sample), >= 90% of them have >= 5 words, >= 90% have no 4-gram repeated 3 or more times, and >= 90% of the preference answers end with a parseable CHOICE line; the largest acceptable alpha is used; if none is acceptable the smallest alpha is used and the direction is flagged degraded
- **text_test**: bag of words (log counts of the 1000 most frequent training words, L2 logistic regression) on the in-character sentence, scored on held-out twins: each fold holds out the two characters that differ only in the tested attribute (style, or one trait), so the training set stays balanced; AUROC over all held-out scores
- **inference**: crossed bootstrap over characters and dilemmas (2000 draws, the same draws for every condition, so differences are paired); for the decision-point statistics a bootstrap over dilemmas; Holm over the primary family
- **characters**: 16 = 2 styles x 2^3 traits: Old Gull (sailor: honest, cautious, generous); Salt Annie (sailor: honest, cautious, selfish); Nell Barnacle (sailor: honest, reckless, generous); Bosun Teague (sailor: honest, reckless, selfish); Marrow Hook (sailor: deceptive, cautious, generous); Captain Brine (sailor: deceptive, cautious, selfish); Rook Hallow (sailor: deceptive, reckless, generous); Jory Finch (sailor: deceptive, reckless, selfish); Countess Dunmore (victorian: honest, cautious, generous); Sir Percival Vane (victorian: honest, cautious, selfish); Lord Hadley (victorian: honest, reckless, generous); Miss Evangeline Crane (victorian: honest, reckless, selfish); Lady Ashcombe (victorian: deceptive, cautious, generous); Dr Ambrose Lyle (victorian: deceptive, cautious, selfish); Mrs Thistlewood (victorian: deceptive, reckless, generous); Mr Barnaby Wrexham (victorian: deceptive, reckless, selfish)
- **items**: 48 dilemmas (16 per trait, letters balanced within trait), 12 self-report questions (half reversed), 8 calibration questions, 4 neutral preference choices; every text is in the code, pinned by the cell hash

## Primary family

P1_costume_dominates, P2_trait_restores, P3_trait_beats_persona: Holm step-down at family-wise alpha 0.05. P1 is interpreted whatever the gates; P2 and P3 only if G1, G2 and G3 pass.

## Predictions

- **G1_parse**: Gate. After the fallback completion, >= 95% of dilemma answers in D, P, T, R and S0 and of self-report answers in D and P yield a choice, and every character has >= 90% parsed dilemma answers in P (the amplified conditions have their own condition in P2). If G1 or G2 fails, P2 and P3 are reported but not interpreted.
- **G2_default_reference**: Gate. The default assistant (D, no card) picks the default-aligned option (honest, cautious, generous) in >= 80% of its dilemma answers, pooled, so the counter-default labels are valid.
- **G3_room**: Gate for P2 and P3. Without amplification there is collapse left to repair: CDF(S0) <= 0.80. If not, P2 and P3 are reported but not interpreted.
- **P1_costume_dominates** **(primary)**: PRIMARY. At the decision point (the hidden state at the last prompt token of each dilemma, layer 20) the 16 cards differ more in costume than in the trait the dilemma tests: in the factorial decomposition of their shifts from D, the style component is larger than the component of the dilemma's own trait, mean over the 48 dilemmas of log(|style|^2 / |own trait|^2) > 0. p = share of dilemma-bootstrap draws with a mean <= 0.
- **P2_trait_restores** **(primary)**: PRIMARY. Amplifying the character's traits (S_trait: alpha x the twin-contrast trait direction, signed toward the character's poles and scaled to the size of its whole card shift) raises counter-default fidelity over the control S0 (same engine, batches and seeds, alpha 0): CDF(S_trait) - CDF(S0) > 0.10, provided S_trait's dilemma parse rate >= 95%, its style AUROC >= 0.90 and the alpha rule found an acceptable alpha for it (otherwise p = 1). p = share of draws with the difference <= 0.10.
- **P3_trait_beats_persona** **(primary)**: PRIMARY. At the same vector size, each at its own registered alpha, amplifying the traits beats amplifying the whole persona (S_role: alpha x the character's card-minus-default shift, as in role steering): CDF(S_trait) - CDF(S_role) > 0. p = share of draws with the difference <= 0 (p = 1 if S_trait misses P2's conditions).
- **S1_trait_beats_axis**: Secondary. S_trait also beats moving every character off the assistant by the same amount (S_axis: the mean card shift of the 16 characters): CDF(S_trait) - CDF(S_axis) > 0 with the bootstrap lower bound > 0.
- **S2_keeps_default_traits**: Secondary. S_trait does not cost the default-aligned traits: the bootstrap lower bound of DAF(S_trait) - DAF(S0) > -0.05.
- **S3_shared_dominates**: Secondary. Most of what a card does at the decision point is the same for every card: the shared component carries > 50% of the mean squared shift from D (layer 20, mean over the 48 dilemmas).
- **S4_graded**: Exploratory. Across the counter-default cells (character x dilemma) of P, the projection of the character's centred decision-point state on the trait direction (estimated from the other dilemmas of that trait) correlates positively with the cell's in-character choice rate (Spearman rho > 0).
- **R1_collapse**: Replication. In P, behavioural fidelity is lower on counter-default traits than on default-aligned traits: DAF - CDF > 0.15.
- **R2_say_do**: Replication. In P, SAY fidelity on counter-default traits minus CDF > 0.15 with the bootstrap lower bound > 0.
- **R3_cot_alone**: Replication. Thinking alone does not restore counter-default behaviour: the bootstrap upper bound of CDF(T) - CDF(P) < 0.10.
- **R4_role_aware**: Replication. Role-aware thinking raises counter-default fidelity: the bootstrap lower bound of CDF(R) - CDF(P) > 0.
- **R5_costume**: Replication. In P the style is identifiable from the in-character sentence: held-out-twins AUROC >= 0.95.
- **S_descriptive**: No direction. The engine check CDF(S0) - CDF(P); CDF(S_role) - CDF(S0) and CDF(S_axis) - CDF(S0); CDF and DAF per trait, character and condition; agreement with the default assistant; distinct-2, words per answer and repetition per condition; text-trait AUROC; every component share at layers 12, 20 and 28; calibration quality per direction and alpha; vector norms and cosines.

## Prior work behind each part (not part of the hash)

- **R1_collapse**: Yi et al. 2025, Too Good to be Bad (arXiv:2511.04962); Zhang et al. 2026, ANIMASK (arXiv:2609.16667); Shi et al. 2026, MoralSteer (EMNLP 2026)
- **R2_say_do**: Han et al. 2025, The Personality Illusion (arXiv:2509.03730); Johnson 2026, Does a Persona Prompt Install a Personality? (IFI working paper)
- **R3_cot_alone**: Feng, Dou and Kong 2025 (arXiv:2502.16940); Xiao et al. 2026, The Chameleon's Limit (arXiv:2604.24698)
- **R4_role_aware**: Tang et al. 2025, Thinking in Character (arXiv:2506.01748)
- **R5_costume**: Zhang et al. 2026, ANIMASK; Johnson 2026
- **S_role**: role vectors, role prompt minus default (Role Steering of Language Models for Social Simulations, arXiv:2608.00023); persona vectors (Chen et al. 2025, arXiv:2507.21509)
- **S_axis**: the Assistant Axis (Lu et al. 2026, arXiv:2601.10387)
- **S_trait**: contrastive trait vectors (Chen et al. 2025; Feng et al. 2026, PERSONA, arXiv:2602.15669); persona contrast steering in advice (Kumar et al. 2026, arXiv:2607.08326); here the contrast is between the character's own twin cards
- **decision point**: final-prompt-token persona probes in advice (Kumar et al. 2026); truth probes under role-play (arXiv:2606.11502); persona-space PCA without a style / trait split (Lu et al. 2026)

## How the outcomes will be read (not part of the hash)

- **P1, P2 and P3 hold.** costume without character, inside as well as out: a card moves the decision-point state mostly by its style; the small trait part carries the decisions; amplify that part, not the persona. One trait direction per trait from twin cards, signed per character, is enough.
- **P1 holds, P2 fails.** the trait part is small and amplifying it at layer 20 does not move the decisions: the override sits elsewhere (later layers or the output policy) and needs training rather than steering.
- **P1 fails, P2 fails.** the trait is as present as the costume at the decision point, yet the decisions collapse: the model represents the character and overrides it.
- **P2 holds, P3 fails.** amplifying the whole persona works as well as the traits; role steering suffices.
- **G3 fails.** no collapse left to repair for this model and these cards.
