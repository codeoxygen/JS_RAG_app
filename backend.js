import express from 'express';
import multer from 'multer';
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

  const app = express();
  const upload = multer({ dest: 'uploads/' });

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 30,
    separators: ["\n", "\n\n", " ", "."],
  });
  
  app.post('/upload', upload.single('file'), async (req, res) => {
    const pdfPath = req.file.path;
    const pdf = await getDocument(pdfPath).promise;

    let text = "";
    const n_pages = pdf.numPages;
    for (let i = 1; i <= n_pages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map((item) => item.str).join(" ");
    }

    pdfText = text; // Store the text content in the global variable

    res.status(200).send('File uploaded and read successfully');
});

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
