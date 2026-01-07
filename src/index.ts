import { Elysia, t } from 'elysia';
import { cors } from '@elysiajs/cors';
import * as ort from 'onnxruntime-node';
import natural from 'natural';
import path from 'path';

// 1. Inisialisasi Tool NLP untuk Preprocessing
const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer;
const stopwords = new Set(natural.stopwords);

/**
 * Fungsi Preprocessing: Membersihkan teks agar sesuai dengan 
 * kebutuhan input model ONNX Anda.
 */
function preprocessText(text: string): string {
    let cleaned = text.toLowerCase()
        .replace(/<[^>]*>?/gm, '')  // Hapus tag HTML
        .replace(/http\S+/g, '')    // Hapus URL
        .replace(/[^a-z\s]/g, '');  // Hapus simbol/angka

    const tokens = tokenizer.tokenize(cleaned) || [];

    // Filter stopwords dan lakukan stemming
    return tokens
        .filter(word => !stopwords.has(word) && word.length > 2)
        .map(word => stemmer.stem(word))
        .join(' ');
}

// 2. Load Model ONNX (Gunakan path absolut untuk Vercel)
const modelPath = path.resolve(process.cwd(), 'model/depression_pipeline.onnx');
const session = await ort.InferenceSession.create(modelPath);

// 3. Definisi Aplikasi Elysia
const app = new Elysia({ prefix: '/api' })
    .use(cors({
        origin: '*', // Sesuaikan dengan domain frontend Vercel Anda nantinya
        methods: ['POST', 'GET', 'OPTIONS']
    }))
    .post('/predict', async ({ body }) => {
        const { text } = body;
        const cleanedText = preprocessText(text);

        // Penanganan input kosong setelah preprocessing
        if (!cleanedText.trim()) {
            return { label: 0, status: 'Stable', note: 'Input too short' };
        }

        // Persiapan Tensor untuk model ONNX
        const inputTensor = new ort.Tensor('string', [cleanedText], [1, 1]);
        const inputName = session.inputNames[0];
        
        const feeds: any = {};
        feeds[inputName] = inputTensor;

        // Jalankan Inference
        const results = await session.run(feeds);
        const outputName = session.outputNames[0];
        const prediction = results[outputName].data[0];

        return {
            label: Number(prediction),
            status: Number(prediction) === 1 ? 'Heavy Heart' : 'Stable', // Label empatis
            processed_text: cleanedText
        };
    }, {
        body: t.Object({
            text: t.String()
        })
    });

// 4. Export Handler untuk Vercel Serverless
export const POST = (req: Request) => app.handle(req);
export const GET = (req: Request) => app.handle(req);

// Tetap gunakan listen untuk testing lokal (Opsional)
app.listen(3001);
console.log(`✨ Senandika Backend (Serverless) ready for Vercel`);
