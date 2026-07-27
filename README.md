# Offline AI Book Reader & Analyzer

A fully offline AI system that reads books (text + images), extracts concepts and relationships, infers new knowledge, reflects on its understanding, sets learning goals, runs curiosity cycles, uses pluggable tools (calculator, file reader, search, OCR, geometry), and answers questions. Optimized for low-end devices with intelligent memory management.

## Features

- **Fully Offline**: No internet connection required
- **Multi-format Support**: PDF, DOCX, TXT, images
- **Learning System**: Extract concepts, relationships, and conclusions from book text
- **Reasoning Engine**: 9 inference rules for chaining facts (A seeks B + B requires C -> A benefits_from C)
- **Reflection Engine**: Detect contradictions, weak concepts, isolated concepts, generate questions
- **Curiosity Engine**: Proactively identifies knowledge gaps, generates questions, seeks answers using tools, and learns from answers
- **Goal System**: Set learning goals, assess knowledge gaps, track progress, auto-complete
- **Pattern Recognizer**: Neural network scores candidate triples, learned relation predictor (A+B+context → relation), semantic similarity, self-training on existing data
- **Tool Framework**: Pluggable tools — Calculator (safe eval), File Reader, Knowledge Graph Search, OCR (EasyOCR), PicoGK geometry stub, and `tool_use()` integration in ReasoningSystem
- **Provenance Tracking**: Every relationship remembers its source book, timestamps, epistemic status, evidence, domain
- **Source Type Layers**: Separate `real`, `fiction`, `speculative` layers with cross-layer contradiction detection, query routing, and truth confidence tracking
- **Image Analysis**: OCR and image understanding with EasyOCR
- **Memory Optimized**: Load-on-demand vectors, memory-mapped database
- **CPU Friendly**: Lightweight embedding models, optional local LLM
- **Smart Compression**: Efficient storage and retrieval

## Architecture

```
Book Input ---> Extraction ---> Embeddings ---> Storage ---> QA
(PDF/DOCX)    (text/img)      (384-dim)      (SQLite)    (concept-aware)

                                                |
                                          Learning System
                                     (concepts/rels/conclusions)
                                                |
                                          Reasoning System
                                        (inference rules)
                                                |
                                          Reflection System
                                      (contradictions/gaps)
                                                |
                                          Curiosity Engine
                                    (identify → ask → seek → learn)
                                                |
                                          Goal System
                                    (progress/auto-complete)

                                          Pattern Recognizer
                                  (neural scorer / relation predictor
                                   A+B+context → relation scores)

                                           Tool Framework
                                    (calculator, search, OCR,
                                     file reader, geometry stub)

                                           Learning Loop
                                    (8-step autonomous cycle)

                                           Web Dashboard
                                    (Flask + vis.js knowledge
                                     graph visualization)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Add Books to Library

```bash
python main.py add-book "path/to/book.pdf"      # Add individual book
python main.py scan-books                         # See what's available
python main.py add-to-folder "path/to/book.pdf"   # Add to books folder
```

### 2. Process All Books (learn + reason + reflect + goal check)

```bash
python main.py process-all                        # Full pipeline
python main.py process-all --no-reason            # Skip reasoning
python main.py process-all --no-reflect           # Skip reflection
python main.py process-all -y                     # Skip confirmation prompt
```

### 3. Learn from Specific Books

```bash
python main.py learn                              # Learn from all books
python main.py learn <book_id>                    # Learn from specific book
```

### 4. Browse What Was Learned

```bash
python main.py show-learning                      # Overview of concepts/relationships
python main.py show-learning --concept politics   # Concept graph
python main.py search-concepts "war"              # Search concepts
```

### 5. Run Reasoning Engine

```bash
python main.py reason                             # Infer new facts from existing relationships
```

### 6. Run Reflection Cycle

```bash
python main.py reflect                            # Find contradictions/weak areas/gaps
python main.py reflect --detail                   # Show detailed lists
```

### 7. Set Learning Goals

```bash
python main.py goals add "Understand politics" --focus politics --priority 5
python main.py goals list                         # List goals
python main.py goals list --status all            # All goals (active+completed)
python main.py goals assess 1                     # Knowledge gap analysis
python main.py goals update 1 --progress 0.5      # Set progress
python main.py goals update 1 --status abandoned  # Change status
python main.py goals complete 1                   # Attempt completion
python main.py goals complete 1 --threshold 0.9   # Custom threshold
```

### 8. Run Curiosity Cycle

```bash
python main.py curiosity run                          # Ask 3 questions about weak/isolated knowledge
python main.py curiosity run --max 5                  # Ask up to 5 questions
```

### 9. Use Tools

```bash
python main.py tools list                              # List all available tools
python main.py tools run calculator expression='2+2'   # Quick math
python main.py tools run file_reader path='README.md'  # Read a file
python main.py tools decide "what is 3*7?"             # Auto-pick best tool
```

### 10. Ask Questions

```bash
python main.py ask "What is the main theme of the book?"               # Default (all layers)
python main.py ask "Who rules the Seven Kingdoms?" --source-type fiction  # Only fiction
python main.py ask "What is lead generation?" --source-type real         # Only business
python main.py chat                               # Interactive mode
python main.py route "Are dragons real?"           # Debug query routing
```

### 9. Run Pattern Recognition (Neural Network)

```bash
python main.py recognize --text "A prince seeks power and controls the state"    # Analyze text
python main.py recognize --book-id 1                   # Analyze a book's text
python main.py recognize --text "..." --threshold 0.15  # Lower threshold for more recall
python main.py train-patterns                           # Train NN on existing relationships
python main.py train-patterns --epochs 50 --lr 0.0005   # Custom training params
python main.py similar-concepts "politics"              # Find semantically similar concepts
python main.py similar-concepts "war" --top-k 5         # Limit results
```

### 9b. Learned Relation Predictor (A + B + context → relation)

```bash
python main.py predict-rel "prince" "power"                                    # Predict relations between concepts
python main.py predict-rel "prince" "power" --context "seeks and controls"      # With context disambiguation
python main.py predict-rel "king" "state" --top-k 5 --threshold 0.2            # Custom display
python main.py train-relation-predictor                                        # Train multi-label classifier
python main.py train-relation-predictor --epochs 100 --lr 0.0005               # Custom training
```

### 11. Run Learning Loop (8-Step Autonomous Cycle)

Runs all learning subsystems in sequence with one command:

```bash
python main.py jafar-loop                         # Full 8-step cycle
python main.py jafar-loop --skip-patterns         # Skip neural patterns
python main.py jafar-loop --skip-reasoning        # Skip inference
python main.py jafar-loop --skip-curiosity        # Skip curiosity
python main.py jafar-loop --skip-decay            # Skip truth decay
python main.py jafar-loop --skip-goals            # Skip goal assessment
python main.py jafar-loop --skip-world            # Skip world model extraction
```

### 12. Launch Web Dashboard

Interactive dashboard with knowledge graph visualization:

```bash
python main.py dashboard                          # http://127.0.0.1:5000
python main.py dashboard --port 8080              # Custom port
python main.py dashboard --debug                  # Debug mode with auto-reload
```

The dashboard provides:
- **Dashboard Overview** — System statistics, truth confidence, source type distribution
- **Knowledge Graph** — Interactive vis.js graph showing concepts (nodes sized by connections) and relationships (colored directed edges)
- **Concept Detail** — Click through to see all relationships for any concept
- **Books** — Searchable library table
- **Goals** — Filterable goals with progress bars
- **Learning Loop** — One-click execution with per-step timing and results

### 13. Manage Library

```bash
python main.py list-books       # List all processed books
python main.py delete-book <id> # Delete a book
python main.py stats            # System statistics
python main.py scan-books       # Scan books folder
python main.py optimize         # Optimize for low-end devices
```

## Configuration

Edit `config.py` to customize:
- Embedding model size
- Memory limits
- Database location
- Processing parameters
- LLM settings (opt-in local model)
- Concept extraction thresholds

## Performance

- **Memory Usage**: ~50-100MB for 1000-page book
- **Processing Speed**: ~10-30 pages/minute
- **Query Response**: <2 seconds
- **Storage**: ~5-10MB per book (compressed)
- **Learning Speed**: ~579 triples from 200-page book in ~5 seconds

## Technical Details

- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2, 384-dim)
- **Vector Search**: FAISS IndexFlatL2 (faiss-cpu 1.14.2)
- **Database**: Shared SQLite (books + vectors + learning tables)
- **OCR**: EasyOCR for image text extraction
- **LLM**: transformers (distilgpt2, opt-in, disabled by default)
- **Memory**: Intelligent caching, load-on-demand vectors
- **Framework**: Python 3.13, Click CLI, CPU-only

## Project Structure

```
ai/
├── main.py                  # CLI entry point (all commands)
├── config.py                # Configuration
├── memory_system.py         # Learning system (4 SQLite tables, provenance)
├── reasoning_system.py      # Inference rules engine (9 rules)
├── reflection_system.py     # Reflection cycle (contradictions/gaps/questions)
├── curiosity_engine.py      # Curiosity engine (identify → ask → seek → learn)
├── goal_system.py           # Learning goals (CRUD, assessment, progress)
├── pattern_recognizer.py    # Neural pattern recognizer (MLP scorer, training, similarity)
├── book_processor.py        # Book processing pipeline
├── book_manager.py          # Books folder management
├── embeddings.py            # Embedding generation
├── vector_db.py             # Vector database (SQLite + FAISS)
├── qa_system.py             # Question answering (concept-aware)
├── memory_manager.py        # Memory optimization
├── utils.py                 # Utilities
├── tools/                   # Pluggable tool framework
│   ├── __init__.py          # default_registry() factory
│   ├── base.py              # Tool ABC + ToolRegistry
│   ├── calculator.py        # Safe math eval with AST visitor
│   ├── file_reader.py       # Local text file reader
│   ├── search.py            # Knowledge graph query tool
│   ├── ocr.py               # Image OCR stub
│   └── picogk.py            # PicoGK geometry kernel stub
├── test_tools.py            # Tool tests (47)
├── test_curiosity.py        # Curiosity tests (25) 
├── test_reasoning.py        # Reasoning tests (5)
├── test_reflection.py       # Reflection tests (5)
├── test_goals.py            # Goals tests (7)
├── test_system.py           # System tests
├── books/                   # User books folder
│   └── README.md
└── data/                    # Database and book storage
    ├── books/
    ├── vectors/
    └── database.db
```
