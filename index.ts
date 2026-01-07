import { Elysia, t } from 'elysia';
import { cors } from '@elysiajs/cors';
import * as ort from 'onnxruntime-node';
import natural from 'natural';

// 1. Inisialisasi Tool NLP
const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer; // Mendekati WordNetLemmatizer di Python
const stopwords = new Set(natural.stopwords); // Mengambil daftar stopword bahasa Inggris standar

function preprocessText(text: string): string {
    // A. Lowercase & Bersihkan Karakter Non-Alfabet (mirip re.sub di Python)
    let cleaned = text.toLowerCase()
        .replace(/<[^>]*>?/gm, '') // Hapus HTML
        .replace(/http\S+/g, '')    // Hapus URL
        .replace(/[^a-z\s]/g, '');  // Hapus angka & simbol

    // B. Tokenization (Memecah kalimat jadi kata)
    const tokens = tokenizer.tokenize(cleaned) || [];

    // C. Stopwords Removal & Stemming
    // Kita filter kata yang terlalu umum, lalu kita sederhanakan (misal: "crying" jadi "cry")
    const processedTokens = tokens
        .filter(word => !stopwords.has(word) && word.length > 2)
        .map(word => stemmer.stem(word));

    return processedTokens.join(' ');
}

// 2. Load Model ONNX
const session = await ort.InferenceSession.create('./model/depression_pipeline.onnx');

const app = new Elysia()
    .use(cors())
    .post('/predict', async ({ body }) => {
        const { text } = body;
        
        // JALANKAN PREPROCESSING BARU
        const cleanedText = preprocessText(text);
        
        // Debugging untuk melihat apa yang "dilihat" oleh model
        console.log(`Input Asli: "${text}"`);
        console.log(`Hasil Preprocessing: "${cleanedText}"`);

        // Jika setelah diproses teks jadi kosong, anggap normal
        if (!cleanedText.trim()) {
            return { label: 0, status: 'Normal', note: 'Input terlalu pendek/umum' };
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
            original_text: text,
            processed_text: cleanedText,
            label: Number(prediction),
            status: Number(prediction) === 1 ? 'Depresi' : 'Normal'
        };
    }, {
        body: t.Object({
            text: t.String()
        })
    })
    .listen(3001);

console.log(`🦊 Backend Hamisfera (Fixed) berjalan di port 3001`);