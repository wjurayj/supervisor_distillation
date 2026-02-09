"""Minimal example: ask a question about a long document using supervisor-worker deep research."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure the package is importable when running from the examples/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill import OpenAIHandler, run

# --- Configuration ---
SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
WORKER_MODEL = os.environ.get("WORKER_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo")
BASE_URL = os.environ.get("BASE_URL", "https://api.together.xyz/v1")
LOG_DIR = os.environ.get("LOG_DIR", "logs/basic_qa")

# --- Build a sample long document ---
SAMPLE_DOC = """
=== Section 1: The History of Computing ===

The history of computing begins long before the modern era of electronic computers.
Early mechanical devices such as the abacus (circa 2400 BCE) and Charles Babbage's
Analytical Engine (1837) laid the groundwork for programmable computation. Ada Lovelace
is widely credited as the first computer programmer for her notes on the Analytical Engine.

In the 1930s and 1940s, Alan Turing formalized the concept of computation with his
theoretical Turing machine, while John von Neumann proposed the stored-program architecture
that underlies virtually all modern computers. The ENIAC (1945) was one of the first
general-purpose electronic computers, weighing 30 tons and occupying 1,800 square feet.

The invention of the transistor at Bell Labs in 1947 by John Bardeen, Walter Brattain,
and William Shockley revolutionized electronics, enabling smaller and faster computers.
The integrated circuit, developed independently by Jack Kilby and Robert Noyce in the
late 1950s, further accelerated miniaturization.

=== Section 2: The Rise of Personal Computing ===

The 1970s saw the emergence of personal computers. The Altair 8800 (1975) is often
considered the first commercially successful personal computer. Apple Computer, founded
by Steve Jobs and Steve Wozniak in 1976, introduced the Apple II in 1977, which became
one of the first mass-produced microcomputers.

IBM entered the personal computer market in 1981 with the IBM PC, which established the
standard architecture that would dominate for decades. Microsoft, founded by Bill Gates
and Paul Allen in 1975, provided the operating system (MS-DOS) for the IBM PC and later
developed Windows, which became the dominant desktop operating system.

The Macintosh, introduced by Apple in 1984, popularized the graphical user interface (GUI)
and the mouse as an input device, concepts originally developed at Xerox PARC.

=== Section 3: The Internet and World Wide Web ===

The Internet originated from ARPANET, a U.S. Department of Defense research project begun
in 1969. The development of TCP/IP protocols in the 1970s by Vint Cerf and Bob Kahn
provided the foundation for a global network of networks.

Tim Berners-Lee invented the World Wide Web in 1989 at CERN, creating HTML, HTTP, and the
first web browser. The release of the Mosaic web browser in 1993 made the web accessible
to the general public. Netscape Navigator, released in 1994, became the dominant web browser
before being overtaken by Microsoft's Internet Explorer in the late 1990s.

The dot-com boom of the late 1990s saw massive investment in Internet-based companies.
Companies like Amazon (founded 1994), Google (founded 1998), and eBay (founded 1995)
emerged during this period and grew to become some of the most valuable companies in the world.

=== Section 4: Mobile Computing and Smartphones ===

The first commercially available smartphone is generally considered to be the IBM Simon
(1994), though the term "smartphone" was not widely used until later. Nokia dominated the
mobile phone market through the late 1990s and early 2000s.

Apple's iPhone, launched in June 2007, revolutionized the smartphone industry with its
touchscreen interface and app ecosystem. Google's Android operating system, first released
in 2008, became the most widely used mobile OS globally.

The App Store model, pioneered by Apple's App Store (2008), created an entirely new software
distribution paradigm. Mobile computing shifted user behavior dramatically, with smartphone
usage eventually exceeding desktop computer usage for web browsing in many markets.

=== Section 5: Artificial Intelligence ===

The field of artificial intelligence was formally founded at the Dartmouth Conference in 1956.
Early AI research focused on symbolic reasoning and expert systems. The "AI winters" of the
1970s and 1980s saw reduced funding and interest due to unmet expectations.

The resurgence of AI began with machine learning approaches, particularly neural networks.
Geoffrey Hinton, Yann LeCun, and Yoshua Bengio are often credited as pioneers of deep learning.
The ImageNet competition in 2012, won by AlexNet (a deep convolutional neural network),
demonstrated the power of deep learning for image recognition.

The introduction of the Transformer architecture by Vaswani et al. in 2017 ("Attention Is All
You Need") revolutionized natural language processing. GPT (Generative Pre-trained Transformer)
models, developed by OpenAI, demonstrated that large language models could perform a wide range
of language tasks. ChatGPT, released in November 2022, brought AI capabilities to mainstream
public awareness.

Recent developments include multimodal AI models capable of processing text, images, and audio;
AI agents that can take actions in software environments; and ongoing research into AI safety,
alignment, and the societal implications of increasingly capable AI systems.
""".strip()

# Repeat the document to make it longer for a more realistic test
LONG_DOC = "\n\n".join([SAMPLE_DOC] * 10)


def main():
    query = "Who invented the World Wide Web and in what year? Also, what was the first web browser released to the general public?"

    print(f"Document length: {len(LONG_DOC):,} chars")
    print(f"Query: {query}")
    print(f"Supervisor: {SUPERVISOR_MODEL}")
    print(f"Worker: {WORKER_MODEL}")
    print(f"Logging to: {LOG_DIR}")
    print("-" * 60)

    supervisor = OpenAIHandler(model=SUPERVISOR_MODEL, base_url=BASE_URL, temperature=0.7, max_tokens=2048)
    worker = OpenAIHandler(model=WORKER_MODEL, base_url=BASE_URL, temperature=0.2, max_tokens=512)

    result = run(
        query=query,
        context=LONG_DOC,
        supervisor=supervisor,
        worker=worker,
        max_iterations=10,
        log_dir=LOG_DIR,
        output_limit=2000,
    )

    print(f"\nAnswer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print(f"Supervisor tokens: {result.supervisor_usage.input_tokens} in / {result.supervisor_usage.output_tokens} out")
    print(f"Worker tokens: {result.worker_usage.input_tokens} in / {result.worker_usage.output_tokens} out")
    print(f"Elapsed: {result.elapsed:.1f}s")


if __name__ == "__main__":
    main()
