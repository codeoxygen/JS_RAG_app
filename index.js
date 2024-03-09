import { config } from "dotenv";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";

import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";

try {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 30,
    separators: ["\n", "\n\n", " ", "."],
  });
  const pdf_path = "./Data/Lahiru Jayakodi CV.pdf";
  getDocument(pdf_path).promise.then(async (pdf) => {
    let text = "";
    const n_pages = pdf.numPages;
    for (let i = 1; i <= n_pages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map((item) => item.str).join(" ");
    }
    config();
    const api_key = process.env.OPENAI_KEY;

    const chunks = await splitter.createDocuments([text]);
    const embedding_model = new OpenAIEmbeddings({
      openAIApiKey: api_key,
    });

    const vectorstore = await FaissStore.fromDocuments(chunks, embedding_model);
    const retriever = vectorstore.asRetriever();

    const llm = new ChatOpenAI({ openAIApiKey: api_key });

    const chain = ConversationalRetrievalQAChain.fromLLM(llm, retriever, {
      memory: new BufferMemory({
        memoryKey: "chat_history",
      }),
    });

    const res = await chain.call({
      question: "What are Lahiru's skills ?",
    });
    console.log(res);

    console.log("Done !");
  });
} catch (err) {
  console.log(err);
}
