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

// Di dalam src/index.ts
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
        
        const inputName = session.inputNames[0];
        if (!inputName) {
            throw new Error('Model input name not found');
        }
        
        const feeds: any = {};
        feeds[inputName] = inputTensor;

        const results = await session.run(feeds);
        
        const outputName = session.outputNames[0];
        if (!outputName) {
            throw new Error('Model output name not found');
        }
        
        const outputTensor = results[outputName];
        if (!outputTensor) {
            throw new Error('Model output tensor not found');
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

// EXPORT UNTUK VERCEL
export const POST = (req: Request) => app.handle(req);
export const GET = (req: Request) => app.handle(req);