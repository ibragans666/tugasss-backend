import { Elysia, t } from 'elysia';
import { cors } from '@elysiajs/cors';
import * as ort from 'onnxruntime-node';
import natural from 'natural';
import path from 'path';

const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer;
const stopwords = new Set(natural.stopwords);

function preprocessText(text: string): string {
    let cleaned = text.toLowerCase()
        .replace(/<[^>]*>?/gm, '') 
        .replace(/http\S+/g, '')   
        .replace(/[^a-z\s]/g, ''); 
    const tokens = tokenizer.tokenize(cleaned) || [];
    return tokens
        .filter(word => !stopwords.has(word) && word.length > 2)
        .map(word => stemmer.stem(word))
        .join(' ');
}

// Menggunakan path absolut untuk model ONNX di Vercel
const modelPath = path.resolve(process.cwd(), 'model/depression_pipeline.onnx');
const session = await ort.InferenceSession.create(modelPath);

const app = new Elysia({ prefix: '/api' })
    .use(cors())
    .post('/predict', async ({ body }) => {
        const { text } = body;
        const cleanedText = preprocessText(text);

        if (!cleanedText.trim()) {
            return { label: 0, status: 'Stable', note: 'Input too short' };
        }

        const inputTensor = new ort.Tensor('string', [cleanedText], [1, 1]);
        
        // SOLUSI TS2538: Pastikan inputName adalah string, bukan undefined
        const inputName = session.inputNames[0];
        if (!inputName) {
            throw new Error('Model input name is undefined');
        }
        
        // Beri tahu TypeScript bahwa objek ini akan menggunakan kunci string
        const feeds: Record<string, ort.Tensor> = {};
        feeds[inputName] = inputTensor;

        const results = await session.run(feeds);
        
        // SOLUSI TS2538: Pastikan outputName adalah string, bukan undefined
        const outputName = session.outputNames[0];
        if (!outputName) {
            throw new Error('Model output name is undefined');
        }
        
        const outputTensor = results[outputName];
        if (!outputTensor) {
            throw new Error('Output tensor not found');
        }

        const prediction = outputTensor.data[0];

        return {
            label: Number(prediction),
            status: Number(prediction) === 1 ? 'Heavy Heart' : 'Stable',
            processed_text: cleanedText
        };
    }, {
        body: t.Object({ text: t.String() })
    });

// Handler untuk Vercel Serverless
export const POST = (req: Request) => app.handle(req);
export const GET = (req: Request) => app.handle(req);

// Opsional: Tetap jalankan listen agar bisa ditest di lokal (localhost:3001)
app.listen(3001);
