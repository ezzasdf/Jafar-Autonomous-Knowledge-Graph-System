#!/usr/bin/env python3
"""
Example usage of the Offline AI Book Reader & Analyzer
Demonstrates how to use the system programmatically
"""

import sys
import os
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
from embeddings import EmbeddingGenerator
from vector_db import VectorDatabase
from book_processor import BookProcessor
from qa_system import QuestionAnsweringSystem
from utils import setup_logging, get_system_info, is_low_end_device

def main():
    """Example usage of the AI book reading system"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Offline AI Book Reader Example")
    
    try:
        # Initialize system components
        logger.info("Initializing system components...")
        
        # Check system capabilities
        system_info = get_system_info()
        is_low_end = is_low_end_device()
        
        logger.info(f"System: {system_info.get('cpu', {}).get('count', 'Unknown')} CPUs, "
                   f"{system_info.get('memory', {}).get('total_gb', 'Unknown'):.1f}GB RAM")
        logger.info(f"Low-end device: {is_low_end}")
        
        # Initialize components
        memory_manager = MemoryManager()
        if is_low_end:
            memory_manager.optimize_for_low_end_device()
        
        embedding_generator = EmbeddingGenerator(memory_manager)
        if is_low_end:
            embedding_generator.optimize_for_low_end_device()
        
        vector_db = VectorDatabase(memory_manager)
        book_processor = BookProcessor(memory_manager)
        qa_system = QuestionAnsweringSystem(embedding_generator, vector_db, memory_manager)
        
        logger.info("System components initialized successfully")
        
        # Example 1: Process a text file
        logger.info("\n=== Example 1: Processing a text file ===")
        
        # Create a sample text file
        sample_text = """
        Artificial Intelligence and Machine Learning
        
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that work and react like humans. Some of the activities 
        computers with artificial intelligence are designed for include speech recognition, 
        learning, planning, and problem solving.
        
        Machine Learning is a subset of AI that provides systems the ability to automatically 
        learn and improve from experience without being explicitly programmed. Machine learning 
        focuses on the development of computer programs that can access data and use it to 
        learn for themselves.
        
        Deep Learning is a subset of machine learning that uses neural networks with multiple 
        layers to model and understand complex patterns. It has been particularly successful 
        in areas like image recognition, natural language processing, and speech recognition.
        
        The future of AI holds great promise for solving complex problems in healthcare, 
        transportation, education, and many other fields. However, it also raises important 
        questions about ethics, privacy, and the future of work.
        """
        
        sample_file = Path("sample_ai_book.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(sample_text)
        
        logger.info(f"Created sample file: {sample_file}")
        
        # Process the book
        book_data = book_processor.process_book(sample_file)
        logger.info(f"Processed book: {book_data['title']}")
        logger.info(f"Pages: {book_data['total_pages']}")
        
        # Add to database
        vector_db.add_book(
            book_data["book_id"],
            book_data["title"],
            book_data["author"],
            str(sample_file),
            book_data["file_type"],
            book_data["total_pages"]
        )
        
        # Process pages and generate embeddings
        for page in book_data["pages"]:
            if page["text_chunks"]:
                texts = [chunk["text"] for chunk in page["text_chunks"]]
                embeddings = embedding_generator.generate_text_embeddings(texts)
                
                for chunk, embedding in zip(page["text_chunks"], embeddings):
                    vector_db.add_page(
                        book_data["book_id"],
                        page["page_number"],
                        chunk["text"],
                        embedding,
                        page["has_images"],
                        page["image_paths"][0] if page["image_paths"] else None
                    )
        
        logger.info("Book added to database successfully")
        
        # Example 2: Ask questions
        logger.info("\n=== Example 2: Asking questions ===")
        
        questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What is deep learning?",
            "What are the applications of AI?",
            "What are the ethical concerns with AI?"
        ]
        
        for question in questions:
            logger.info(f"\nQuestion: {question}")
            
            answer = qa_system.answer_question(question, book_data["book_id"])
            
            logger.info(f"Answer: {answer['answer']}")
            logger.info(f"Confidence: {answer['confidence']:.2f}")
            logger.info(f"Sources: {len(answer['sources'])} passages")
            
            if answer.get("extracted_concepts"):
                concepts_str = ", ".join(c["concept"] for c in answer["extracted_concepts"][:5])
                logger.info(f"Concepts: {concepts_str}")
            
            if answer["sources"]:
                for i, source in enumerate(answer["sources"][:2], 1):
                    logger.info(f"  Source {i}: Page {source['page_number']} "
                              f"(similarity: {source['similarity']:.2f})")
        
        # Example 3: Interactive chat
        logger.info("\n=== Example 3: Interactive chat ===")
        
        # Simulate a conversation
        conversation = [
            ("What is the relationship between AI and machine learning?", None),
            ("Can you explain more about neural networks?", "it"),
            ("What are some real-world applications?", "that")
        ]
        
        previous_question = None
        previous_answer = None
        
        for question, follow_up_indicator in conversation:
            logger.info(f"\nUser: {question}")
            
            if follow_up_indicator and previous_question and previous_answer:
                answer = qa_system.ask_follow_up_question(
                    previous_question, previous_answer["answer"], question
                )
            else:
                answer = qa_system.answer_question(question, book_data["book_id"])
            
            logger.info(f"AI: {answer['answer']}")
            logger.info(f"Confidence: {answer['confidence']:.2f}")
            
            previous_question = question
            previous_answer = answer
        
        # Example 4: System statistics
        logger.info("\n=== Example 4: System statistics ===")
        
        db_stats = vector_db.get_database_stats()
        memory_stats = memory_manager.get_memory_stats()
        
        logger.info(f"Database stats:")
        logger.info(f"  Books: {db_stats.get('book_count', 0)}")
        logger.info(f"  Pages: {db_stats.get('page_count', 0)}")
        logger.info(f"  Database size: {db_stats.get('database_size_mb', 0):.2f}MB")
        
        logger.info(f"Memory stats:")
        logger.info(f"  Current usage: {memory_stats.get('current_memory_mb', 0):.1f}MB")
        logger.info(f"  Memory limit: {memory_stats.get('max_memory_mb', 0):.1f}MB")
        logger.info(f"  Embedding cache: {memory_stats.get('embedding_cache_size', 0)} items")
        
        # Cleanup
        logger.info("\n=== Cleanup ===")
        
        # Delete the sample file
        if sample_file.exists():
            sample_file.unlink()
            logger.info("Sample file deleted")
        
        # Cleanup system resources
        qa_system.cleanup()
        vector_db.cleanup()
        embedding_generator.cleanup()
        memory_manager.cleanup()
        
        logger.info("System cleanup complete")
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 