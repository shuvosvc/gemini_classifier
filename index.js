const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');
const jwt = require('jsonwebtoken');
const fs = require('fs/promises');
const database = require('./utils/connection');
const { port, jwtSecret, geminiApiKey } = require('./config/ApplicationSettings');
const { authintication, authfilereq } = require('./utils/common'); // Import the utility function
const { log } = require('console');


// Import Google Generative AI SDK
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');



const app = express();


const cors = require('cors');




app.use(cors());


app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/profiles', express.static(path.join(__dirname, 'profiles')));

// app.use('/uploads', authfilereq, express.static(path.join(__dirname, 'uploads')));
// app.use('/profiles', authfilereq, express.static(path.join(__dirname, 'profiles')));


app.get("/", (_, res) => {
    res.send("🟢 Server is alive");
});



// --- Gemini API Configuration ---

const genAI = new GoogleGenerativeAI(geminiApiKey);
// Using gemini-1.5-flash for faster, cost-effective multimodal capabilities
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });




// Multer setup
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: {
        files: 10, // Maximum number of files
        fileSize: 10 * 1024 * 1024, // Maximum file size (e.g., 10 MB per file) - Increased for prescriptions/reports
    },
    fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png|webp/; // Allow these image types
        const fileExt = path.extname(file.originalname).toLowerCase(); // Get extension safely
        const isValidType = allowedTypes.test(fileExt);
        const isValidMime = allowedTypes.test(file.mimetype); // Also check MIME type

        if (isValidType && isValidMime) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only jpeg, jpg, png, webp files are allowed.'));
        }
    },
});

// --- Helper function to convert buffer to a suitable format for Gemini Vision ---
// This function returns the 'inlineData' structure directly.
function fileToGenerativePart(buffer, mimeType) {
    return {
        inlineData: {
            data: buffer.toString('base64'),
            mimeType
        },
    };
}

// --- Image Processing Function ---
// This function prepares images for storage and for AI analysis.
const processImages = async (files, userId) => {
    const processedImages = [];
    for (const file of files) {
        const imageBuffer = file.buffer; // Original uploaded image buffer
        const mimeType = file.mimetype;  // Original MIME type from Multer

        const currentDate = new Date().toISOString().replace(/[-:]/g, '');

        const originalSharp = sharp(imageBuffer);

        // Create color image (always converted to PNG for consistency in storage)
        const colorBuffer = await originalSharp
            .toFormat('png')
            .toBuffer();

        // Create thumbnail image (always converted to PNG for consistency in storage)
        const thumbnailBuffer = await originalSharp
            .resize(200, 200, { fit: 'inside' })
            .toFormat('png')
            .toBuffer();

        // Generate consistent filenames (removing original extension, adding .png)
        const baseName = path.basename(file.originalname, path.extname(file.originalname));
        const colorFilename = `${baseName}-color-${userId}-${currentDate}.png`;
        const thumbnailFilename = `${baseName}-thumbnail-${userId}-${currentDate}.png`;



        processedImages.push({
            originalBufferForAI: imageBuffer,    // Original buffer for Gemini
            originalMimeTypeForAI: mimeType,     // Original MIME type for Gemini
            color: { buffer: colorBuffer, filename: colorFilename },
            thumbnail: { buffer: thumbnailBuffer, filename: thumbnailFilename },
        });
    }
    return processedImages;
};

// --- Unified Gemini Classification and Data Extraction Function ---
// RENAMED for clarity and unification
const classifyDocumentAndExtractData = async (imageBuffer, mimeType) => {
    try {
        const prompt = `Analyze the uploaded image and classify its primary document type. If it's a medical document, extract relevant information.

        **Document Types:**
        1.  **"prescription":** A written order from a doctor or healthcare professional for a patient to receive specific medication, treatment, or medical device. Key features usually include patient's name, doctor's name/signature, date, medication details (name, dosage, instructions), clinic/hospital info.
        2.  **"report":** A medical test result or diagnostic report (e.g., blood test, urine test, X-ray report, MRI report, pathology report). Key features usually include patient's name, test name, result values, reference ranges, reporting date.
        3.  **"other":** Any other document, medical or non-medical, that does not fit the "prescription" or "report" categories (e.g., bill, insurance document, discharge summary, appointment slip, personal photo).

        **For "prescription" type, extract:**
        -   **documentType:** "prescription"
        -   **department:** Medical department (string | null). Infer if not explicit.
        -   **doctor_name:** Full name of the prescribing doctor (string | null).
        -   **visited_date:** Date of visit/issue (YYYY-MM-DD | null).

        **For "report" type, extract:**
        -   **documentType:** "report"
        -   **test_name:** The primary name of the test or type of report (string | null, e.g., "Complete Blood Count", "Blood Glucose", "X-Ray Chest").
        -   **deliveryDate:** The date the report was issued or delivered (YYYY-MM-DD | null).
        -   **normal_or_not:** Determine if the *overall* findings or *key results* of the report are within normal limits based on the provided reference ranges. Return "Normal", "Abnormal", or "Not Applicable" (e.g., for X-rays where numerical ranges aren't typical, or if determination is ambiguous).

        **For "other" type:**
        -   **documentType:** "other"
        -   All other fields should be null.

        Return the output as a JSON object with the following structure:
        {
          "documentType": "prescription" | "report" | "other",
          "extractedData": {
            "department": string | null,
            "doctor_name": string | null,
            "visited_date": string | null,
            "test_name": string | null,
            "deliveryDate": string | null,
            "normal_or_not": "Normal" | "Abnormal" | "Not Applicable" | null
          },
          "reason": string | null // Optional: A brief reason if "other", or "classified as [type]"
        }
        `;



        const contents = [{
            role: 'user',
            parts: [
                fileToGenerativePart(imageBuffer, mimeType), // returns { inlineData: {...} }
                { text: prompt }
            ]
        }];


        const result = await model.generateContent({
            contents,
            generationConfig: {
                temperature: 0.1, // Low temperature for factual extraction
                responseMimeType: "application/json", // Request JSON output
            },
            safetySettings: [
                { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
            ],
        });

        const responseText = result.response.text();
        const parsedResult = JSON.parse(responseText);

        console.log("Gemini Classification Result:", JSON.stringify(parsedResult, null, 2));

        // Basic validation of Gemini's response structure
        if (!parsedResult || !parsedResult.documentType || !parsedResult.extractedData) {
            console.warn("Gemini returned unexpected structure. Falling back to 'other'.");
            return {
                documentType: "other",
                extractedData: {
                    department: null, doctor_name: null, visited_date: null,
                    test_name: null, deliveryDate: null, normal_or_not: null
                },
                reason: 'AI classification result malformed.'
            };
        }

        return parsedResult;

    } catch (error) {
        console.error('Error during Gemini classification and data extraction:', error);
        return {
            documentType: "other", // Default to 'other' on error
            extractedData: {
                department: null, doctor_name: null, visited_date: null,
                test_name: null, deliveryDate: null, normal_or_not: null
            },
            reason: 'Error during AI processing or API call failed.'
        };
    }
};





app.post('/uploadPrescription', upload.array('image'), async (req, res) => {
    const connection = await database.getConnection();
    try {
        let {
            accessToken,
            member_id,
            department,
            doctor_name,
            visited_date,
            title,
            shared
        } = req.body;

        // Validate required fields
        if (!member_id || !Number.isInteger(+member_id) || +member_id <= 0) {
            return res.status(400).json({ error: 'Invalid or missing member_id' });
        }
        // if (!title || typeof title !== 'string' || title.trim().length === 0) {
        //     return res.status(400).json({ error: 'Missing or invalid prescription title' });
        // }

        // Optional validations (still useful for direct user input)
        if (department !== undefined && typeof department !== 'string') {
            return res.status(400).json({ error: 'department must be a string' });
        }
        if (shared !== undefined && shared !== 'true' && shared !== 'false' && shared !== true && shared !== false) {
            return res.status(400).json({ error: 'Invalid shared value. Must be true or false' });
        }
        if (doctor_name !== undefined && typeof doctor_name !== 'string') {
            return res.status(400).json({ error: 'doctor_name must be a string' });
        }
        if (visited_date !== undefined) {
            const visitDateParsed = new Date(visited_date);
            if (isNaN(visitDateParsed.getTime())) {
                return res.status(400).json({ error: 'Invalid visited_date format. Use YYYY-MM-DD' });
            }
        }

        const { decodedToken } = await authintication(accessToken, member_id, connection);

        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No files uploaded.' });
        }

        await connection.beginTransaction();

        const uploadFolder = path.join(__dirname, 'uploads');
        await fs.mkdir(uploadFolder, { recursive: true });

        const processedImages = await processImages(req.files, decodedToken.userId);
        const invalidFiles = []; // Files that are not classified as 'prescription'
        const validPrescriptionImages = []; // Images that ARE classified as 'prescription'
        const extractedDataFromImages = {
            department: new Set(),
            doctor_name: new Set(),
            visited_date: new Set()
        };

        for (let i = 0; i < processedImages.length; i++) {
            const { originalBufferForAI, originalMimeTypeForAI } = processedImages[i]; // No need for color, thumbnail here
            // !!! IMPORTANT: Call the UNIFIED function here !!!
            const classificationResult = await classifyDocumentAndExtractData(originalBufferForAI, originalMimeTypeForAI);

            if (classificationResult.documentType !== 'prescription') { // Check against new documentType
                invalidFiles.push({
                    index: i,
                    originalName: req.files[i].originalname,
                    classifiedAs: classificationResult.documentType,
                    reason: classificationResult.reason || `Document classified as '${classificationResult.documentType}', not a prescription.`
                });
                continue; // Skip this file
            }

            // Collect extracted data if valid and available
            if (classificationResult.extractedData) {
                if (classificationResult.extractedData.department) extractedDataFromImages.department.add(classificationResult.extractedData.department);
                if (classificationResult.extractedData.doctor_name) extractedDataFromImages.doctor_name.add(classificationResult.extractedData.doctor_name);
                if (classificationResult.extractedData.visited_date) extractedDataFromImages.visited_date.add(classificationResult.extractedData.visited_date);
            }
            validPrescriptionImages.push(processedImages[i]); // Add to valid images for later storage
        }

        // If any file was not a prescription, reject the whole batch for this endpoint
        if (invalidFiles.length > 0) {
            await connection.rollback();
            return res.status(400).json({
                error: 'Some uploaded files are not recognized as medical prescriptions. All files were rejected.',
                invalidFiles
            });
        }

        // Auto-fill logic for prescription details
        if (department === undefined && extractedDataFromImages.department.size === 1) {
            department = [...extractedDataFromImages.department][0];
        }
        if (doctor_name === undefined && extractedDataFromImages.doctor_name.size === 1) {
            doctor_name = [...extractedDataFromImages.doctor_name][0];
        }
        if (visited_date === undefined && extractedDataFromImages.visited_date.size === 1) {
            visited_date = [...extractedDataFromImages.visited_date][0];
        }

        // Build dynamic INSERT query for the prescriptions table
        const fields = ['user_id'];
        const values = [member_id];
        const placeholders = ['$1'];
        let idx = 2;

        if (title !== undefined) {
            fields.push('title');
            values.push(title);
            placeholders.push(`$${idx++}`);
        }
        if (department !== undefined) {
            fields.push('department');
            values.push(department);
            placeholders.push(`$${idx++}`);
        }
        if (shared !== undefined) {
            fields.push('shared');
            values.push(shared === 'true' || shared === true);
            placeholders.push(`$${idx++}`);
        }
        if (doctor_name !== undefined) {
            fields.push('doctor_name');
            values.push(doctor_name);
            placeholders.push(`$${idx++}`);
        }
        if (visited_date !== undefined) {
            fields.push('visited_date');
            values.push(visited_date);
            placeholders.push(`$${idx++}`);
        }
        fields.push('created_at');
        placeholders.push('CURRENT_DATE');

        const insertPrescriptionQuery = `
            INSERT INTO prescriptions (${fields.join(', ')})
            VALUES (${placeholders.join(', ')})
            RETURNING id
        `;

        const { id: prescriptionId } = await connection.queryOne(insertPrescriptionQuery, values);

        // Write files and insert image records for valid prescription images
        for (const img of validPrescriptionImages) {
            const colorPath = path.join(uploadFolder, img.color.filename);
            const thumbPath = path.join(uploadFolder, img.thumbnail.filename);
            await fs.writeFile(colorPath, img.color.buffer);
            await fs.writeFile(thumbPath, img.thumbnail.buffer);

            await connection.query(
                `INSERT INTO prescription_images (prescription_id, resiged, thumb, created_at)
                 VALUES ($1, $2, $3, CURRENT_DATE)`,
                [prescriptionId, `/uploads/${img.color.filename}`, `/uploads/${img.thumbnail.filename}`]
            );
        }

        await connection.commit();
        res.status(200).json({
            message: 'Prescription uploaded successfully.',
            prescriptionId,
            autoFilledData: {
                department: department || null,
                doctor_name: doctor_name || null,
                visited_date: visited_date || null
            }
        });

    } catch (error) {
        await connection.rollback();

        if (error instanceof multer.MulterError) {
            if (error.code === 'LIMIT_FILE_COUNT') {
                return res.status(400).json({ error: 'You can only upload a maximum of 10 files.' });
            } else if (error.code === 'LIMIT_FILE_SIZE') {
                return res.status(400).json({ error: 'File size exceeds the limit (10MB per file).' });
            }
        }

        if (['Access token is required.', 'Invalid or expired access token.', 'Invalid user.'].includes(error.message)) {
            return res.status(403).json({ error: error.message });
        }

        console.error("Error uploading prescription:", error);
        return res.status(500).json({ error: 'An unknown error occurred. Please try again later.' });
    } finally {
        await connection.release();
    }
});


// Route to append images to an existing prescription
app.post('/appendPrescriptionImages', upload.array('image'), async (req, res) => {
    const connection = await database.getConnection();
    try {
        const { accessToken, member_id, prescription_id } = req.body;

        if (!member_id || !Number.isInteger(+member_id) || +member_id <= 0) {
            return res.status(400).json({ error: 'Invalid member_id' });
        }
        if (!prescription_id || !Number.isInteger(+prescription_id) || +prescription_id <= 0) {
            return res.status(400).json({ error: 'Invalid prescription_id' });
        }
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No images uploaded.' });
        }

        const { decodedToken } = await authintication(accessToken, member_id, connection);

        const prescription = await connection.queryOne(
            'SELECT user_id FROM prescriptions WHERE id = $1 AND deleted = false',
            [prescription_id]
        );
        if (!prescription || prescription.user_id !== +member_id) {
            return res.status(403).json({ error: 'Prescription not found or does not belong to this member.' });
        }

        await connection.beginTransaction();

        const uploadFolder = path.join(__dirname, 'uploads');
        await fs.mkdir(uploadFolder, { recursive: true });

        const processedImages = await processImages(req.files, decodedToken.userId);
        const invalidFiles = []; // Files that are not classified as 'prescription'
        const validPrescriptionImages = [];

        for (let i = 0; i < processedImages.length; i++) {
            const { originalBufferForAI, originalMimeTypeForAI } = processedImages[i];
            // !!! IMPORTANT: Call the UNIFIED function here !!!
            const classificationResult = await classifyDocumentAndExtractData(originalBufferForAI, originalMimeTypeForAI);

            if (classificationResult.documentType !== 'prescription') { // Check against new documentType
                invalidFiles.push({
                    index: i,
                    originalName: req.files[i].originalname,
                    classifiedAs: classificationResult.documentType,
                    reason: classificationResult.reason || `Document classified as '${classificationResult.documentType}', not a prescription.`
                });
                continue;
            }
            validPrescriptionImages.push(processedImages[i]);
        }

        if (invalidFiles.length > 0) {
            await connection.rollback();
            return res.status(400).json({ error: 'Some images are not medical prescriptions and were not appended.', invalidFiles });
        }

        for (const img of validPrescriptionImages) {
            const colorPath = path.join(uploadFolder, img.color.filename);
            const thumbPath = path.join(uploadFolder, img.thumbnail.filename);
            await fs.writeFile(colorPath, img.color.buffer);
            await fs.writeFile(thumbPath, img.thumbnail.buffer);

            await connection.query(
                `INSERT INTO prescription_images (prescription_id, resiged, thumb, created_at)
                 VALUES ($1, $2, $3, CURRENT_DATE)`,
                [prescription_id, `/uploads/${img.color.filename}`, `/uploads/${img.thumbnail.filename}`]
            );
        }

        await connection.commit();
        res.status(200).json({ message: 'Images appended successfully to prescription.' });

    } catch (error) {
        await connection.rollback();
        console.error("Error appending prescription images:", error);
        if (error instanceof multer.MulterError) {
            if (error.code === 'LIMIT_FILE_COUNT') {
                return res.status(400).json({ error: 'You can only upload a maximum of 10 files.' });
            } else if (error.code === 'LIMIT_FILE_SIZE') {
                return res.status(400).json({ error: 'File size exceeds the limit (10MB per file).' });
            }
        }
        if (['Access token is required.', 'Invalid or expired access token.', 'Invalid user.', 'Prescription not found or does not belong to this member.'].includes(error.message)) {
            return res.status(403).json({ error: error.message });
        }
        return res.status(500).json({ error: 'An unknown error occurred. Please try again later.' });
    } finally {
        await connection.release();
    }
});


// --- NEW Route: Upload Medical Reports ---
app.post('/uploadReport', upload.array('image'), async (req, res) => {
    const connection = await database.getConnection();
    try {
        let {
            accessToken,
            member_id,
            prescription_id,
            test_name,       // User provided, will be auto-filled if undefined and extracted
            deliveryDate,    // User provided, will be auto-filled if undefined and extracted
            normal_or_not,   // User provided, will be auto-filled if undefined and extracted
            title,
            shared
        } = req.body;

        // Validate required fields
        if (!member_id || !Number.isInteger(+member_id) || +member_id <= 0) {
            return res.status(400).json({ error: 'Invalid or missing member_id' });
        }

        // Validate required title
        if (title !== undefined && (typeof title !== 'string' || title.trim().length === 0)) {
            return res.status(400).json({ error: 'Invalid report title' });
        }


        // Optional validations
        if (prescription_id !== undefined && (!Number.isInteger(+prescription_id) || +prescription_id <= 0)) {
            return res.status(400).json({ error: 'Invalid prescription_id' });
        }

        if (prescription_id !== undefined) {
            const prescription = await connection.queryOne(
                `SELECT user_id FROM prescriptions WHERE id = $1 AND deleted = false`,
                [prescription_id]
            );

            if (!prescription) {
                return res.status(404).json({ error: 'Prescription not found.' });
            }

            if (prescription.user_id !== +member_id) {
                return res.status(403).json({ error: 'Unauthorized. Prescription does not belong to this member.' });
            }
        }


        // Optional validations for report-specific fields
        if (test_name !== undefined && typeof test_name !== 'string') {
            return res.status(400).json({ error: 'test_name must be a string' });
        }
        if (deliveryDate !== undefined) {
            const dateParsed = new Date(deliveryDate);
            if (isNaN(dateParsed.getTime())) {
                return res.status(400).json({ error: 'Invalid deliveryDate format. Use YYYY-MM-DD' });
            }
        }
        // Validate normal_or_not if provided by user
        const allowedNormalOrNot = ['Normal', 'Abnormal', 'null', null, '']; // Include string 'null' and empty string for flexibility
        if (normal_or_not !== undefined && !allowedNormalOrNot.includes(normal_or_not)) {
            return res.status(400).json({ error: 'Invalid normal_or_not value. Must be "Normal", "Abnormal"or empty.' });
        }
        // Convert user provided normal_or_not to a standardized format if it's a string 'null' or empty
        if (normal_or_not === 'null' || normal_or_not === '') {
            normal_or_not = null;
        }
        if (shared !== undefined && shared !== 'true' && shared !== 'false' && shared !== true && shared !== false) {
            return res.status(400).json({ error: 'Invalid shared value. Must be true or false' });
        }

        const { decodedToken } = await authintication(accessToken, member_id, connection);

        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No files uploaded.' });
        }

        await connection.beginTransaction();

        const uploadFolder = path.join(__dirname, 'uploads');
        await fs.mkdir(uploadFolder, { recursive: true });

        const processedImages = await processImages(req.files, decodedToken.userId);
        const invalidFiles = []; // Files that are not classified as 'report'
        const validReportImages = []; // Images that ARE classified as 'report'
        const extractedDataFromImages = {
            test_name: new Set(),
            deliveryDate: new Set(),
            normal_or_not: new Set()
        };

        for (let i = 0; i < processedImages.length; i++) {
            const { originalBufferForAI, originalMimeTypeForAI } = processedImages[i];
            // !!! IMPORTANT: Call the UNIFIED function here !!!
            const classificationResult = await classifyDocumentAndExtractData(originalBufferForAI, originalMimeTypeForAI);

            if (classificationResult.documentType !== 'report') { // Check against new documentType
                invalidFiles.push({
                    index: i,
                    originalName: req.files[i].originalname,
                    classifiedAs: classificationResult.documentType,
                    reason: classificationResult.reason || `Document classified as '${classificationResult.documentType}', not a medical report.`
                });
                continue;
            }

            // Collect extracted data if valid and available
            if (classificationResult.extractedData) {
                if (classificationResult.extractedData.test_name) extractedDataFromImages.test_name.add(classificationResult.extractedData.test_name);
                if (classificationResult.extractedData.deliveryDate) extractedDataFromImages.deliveryDate.add(classificationResult.extractedData.deliveryDate);
                if (classificationResult.extractedData.normal_or_not) extractedDataFromImages.normal_or_not.add(classificationResult.extractedData.normal_or_not);
            }
            validReportImages.push(processedImages[i]);
        }

        if (invalidFiles.length > 0) {
            await connection.rollback();
            return res.status(400).json({
                error: 'Some uploaded files are not recognized as medical reports. All files were rejected.',
                invalidFiles
            });
        }

        // Auto-fill logic for report details
        if (test_name === undefined && extractedDataFromImages.test_name.size === 1) {
            test_name = [...extractedDataFromImages.test_name][0];
        }
        if (deliveryDate === undefined && extractedDataFromImages.deliveryDate.size === 1) {
            deliveryDate = [...extractedDataFromImages.deliveryDate][0];
        }
        if (normal_or_not === undefined && extractedDataFromImages.normal_or_not.size === 1) {
            normal_or_not = [...extractedDataFromImages.normal_or_not][0];
        }

        // Build dynamic INSERT query for the reports table (You'll need a 'reports' table in your DB)
        const fields = ['user_id', 'title'];
        const values = [member_id, title];
        const placeholders = ['$1', '$2'];
        let idx = 3;


        if (prescription_id !== undefined) {
            fields.push('prescription_id');
            values.push(prescription_id);
            placeholders.push(`$${idx++}`);
        }

        if (test_name !== undefined) {
            fields.push('test_name');
            values.push(test_name);
            placeholders.push(`$${idx++}`);
        }
        if (deliveryDate !== undefined) {
            fields.push('delivery_date'); // Ensure your DB column matches this casing
            values.push(deliveryDate);
            placeholders.push(`$${idx++}`);
        }
        if (normal_or_not !== undefined) {
            fields.push('normal_or_not'); // Ensure your DB column matches this casing
            values.push(normal_or_not);
            placeholders.push(`$${idx++}`);
        }
        if (shared !== undefined) {
            fields.push('shared');
            values.push(shared === 'true' || shared === true);
            placeholders.push(`$${idx++}`);
        }
        fields.push('created_at');
        placeholders.push('CURRENT_DATE');

        const insertReportQuery = `
            INSERT INTO reports (${fields.join(', ')})
            VALUES (${placeholders.join(', ')})
            RETURNING id
        `;

        const { id: reportId } = await connection.queryOne(insertReportQuery, values);

        // Write files and insert image records for valid report images (You'll need a 'report_images' table)
        for (const img of validReportImages) {
            const colorPath = path.join(uploadFolder, img.color.filename);
            const thumbPath = path.join(uploadFolder, img.thumbnail.filename);
            await fs.writeFile(colorPath, img.color.buffer);
            await fs.writeFile(thumbPath, img.thumbnail.buffer);

            await connection.query(
                `INSERT INTO report_images (report_id, resiged, thumb, created_at)
                 VALUES ($1, $2, $3, CURRENT_DATE)`,
                [reportId, `/uploads/${img.color.filename}`, `/uploads/${img.thumbnail.filename}`]
            );
        }

        await connection.commit();
        res.status(200).json({
            message: 'Report uploaded successfully.',
            reportId,
            autoFilledData: {
                test_name: test_name || null,
                deliveryDate: deliveryDate || null,
                normal_or_not: normal_or_not || null
            }
        });

    } catch (error) {
        await connection.rollback();

        if (error instanceof multer.MulterError) {
            if (error.code === 'LIMIT_FILE_COUNT') {
                return res.status(400).json({ error: 'You can only upload a maximum of 10 files.' });
            } else if (error.code === 'LIMIT_FILE_SIZE') {
                return res.status(400).json({ error: 'File size exceeds the limit (10MB per file).' });
            }
        }

        if (['Access token is required.', 'Invalid or expired access token.', 'Invalid user.'].includes(error.message)) {
            return res.status(403).json({ error: error.message });
        }

        console.error("Error uploading report:", error);
        return res.status(500).json({ error: 'An unknown error occurred. Please try again later.' });
    } finally {
        await connection.release();
    }
});


// --- NEW Route: Append Images to an Existing Report ---
app.post('/appendReportImages', upload.array('image'), async (req, res) => {
    const connection = await database.getConnection();
    try {
        const { accessToken, member_id, report_id } = req.body;

        if (!member_id || !Number.isInteger(+member_id) || +member_id <= 0) {
            return res.status(400).json({ error: 'Invalid member_id' });
        }
        if (!report_id || !Number.isInteger(+report_id) || +report_id <= 0) {
            return res.status(400).json({ error: 'Invalid report_id' });
        }
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No images uploaded.' });
        }

        const { decodedToken } = await authintication(accessToken, member_id, connection);

        const report = await connection.queryOne(
            'SELECT user_id FROM reports WHERE id = $1 AND deleted = false',
            [report_id]
        );
        if (!report || report.user_id !== +member_id) {
            return res.status(403).json({ error: 'Report not found or does not belong to this member.' });
        }

        await connection.beginTransaction();

        const uploadFolder = path.join(__dirname, 'uploads');
        await fs.mkdir(uploadFolder, { recursive: true });

        const processedImages = await processImages(req.files, decodedToken.userId);
        const invalidFiles = []; // Files that are not classified as 'report'
        const validReportImages = [];

        for (let i = 0; i < processedImages.length; i++) {
            const { originalBufferForAI, originalMimeTypeForAI } = processedImages[i];
            // !!! IMPORTANT: Call the UNIFIED function here !!!
            const classificationResult = await classifyDocumentAndExtractData(originalBufferForAI, originalMimeTypeForAI);

            if (classificationResult.documentType !== 'report') { // Check against new documentType
                invalidFiles.push({
                    index: i,
                    originalName: req.files[i].originalname,
                    classifiedAs: classificationResult.documentType,
                    reason: classificationResult.reason || `Document classified as '${classificationResult.documentType}', not a medical report.`
                });
                continue;
            }
            validReportImages.push(processedImages[i]);
        }

        if (invalidFiles.length > 0) {
            await connection.rollback();
            return res.status(400).json({ error: 'Some images are not medical reports and were not appended.', invalidFiles });
        }

        for (const img of validReportImages) {
            const colorPath = path.join(uploadFolder, img.color.filename);
            const thumbPath = path.join(uploadFolder, img.thumbnail.filename);
            await fs.writeFile(colorPath, img.color.buffer);
            await fs.writeFile(thumbPath, img.thumbnail.buffer);

            await connection.query(
                `INSERT INTO report_images (report_id, resiged, thumb, created_at)
                 VALUES ($1, $2, $3, CURRENT_DATE)`,
                [report_id, `/uploads/${img.color.filename}`, `/uploads/${img.thumbnail.filename}`]
            );
        }

        await connection.commit();
        res.status(200).json({ message: 'Images appended successfully to report.' });

    } catch (error) {
        await connection.rollback();
        console.error("Error appending report images:", error);

        if (error instanceof multer.MulterError) {
            if (error.code === 'LIMIT_FILE_COUNT') {
                return res.status(400).json({ error: 'You can only upload a maximum of 10 files.' });
            } else if (error.code === 'LIMIT_FILE_SIZE') {
                return res.status(400).json({ error: 'File size exceeds the limit (10MB per file).' });
            }
        }
        if (['Access token is required.', 'Invalid or expired access token.', 'Invalid user.', 'Report not found or does not belong to this member.'].includes(error.message)) {
            return res.status(403).json({ error: error.message });
        }
        return res.status(500).json({ error: 'An unknown error occurred. Please try again later.' });
    } finally {
        await connection.release();
    }
});

























































app.get('/getSharedDocs', async (req, res) => {
    const { token } = req.query;

    if (!token || typeof token !== 'string') {
        return res.status(400).json({ error: 'Token is required' });
    }

    const connection = await database.getConnection();
    try {
        // Step 1: Validate token
        const tokenInfo = await connection.queryOne(
            `SELECT user_id, expires_at FROM token WHERE token = $1`,
            [token]
        );

        if (!tokenInfo) {
            return res.status(403).json({ error: 'Invalid token' });
        }

        const { user_id, expires_at } = tokenInfo;
        const now = new Date();
        const expiry = new Date(expires_at);

        if (expiry < now) {
            return res.status(403).json({ error: 'Token has expired' });
        }

        const expiresInSeconds = Math.floor((expiry.getTime() - now.getTime()) / 1000);
        if (expiresInSeconds <= 0) {
            return res.status(403).json({ error: 'Token has expired' });
        }

        // Step 2: Generate JWT for file access
        const fileAccessToken = jwt.sign({ userId: user_id }, jwtSecret, {
            expiresIn: expiresInSeconds
        });

        // Step 3: Fetch shared prescriptions
        const prescriptions = await connection.query(
            `SELECT id, title, department, doctor_name, visited_date, created_at
       FROM prescriptions
       WHERE user_id = $1 AND shared = true AND deleted = false
       ORDER BY created_at DESC`,
            [user_id]
        );

        const prescriptionIds = prescriptions.map(p => p.id);

        // Step 4: Fetch prescription images
        const prescriptionImages = await connection.query(
            `SELECT prescription_id, id as prescription_img_id, resiged, thumb
       FROM prescription_images
       WHERE prescription_id = ANY($1::int[]) AND deleted = false
       ORDER BY created_at ASC`,
            [prescriptionIds]
        );

        const prescriptionImageMap = {};
        for (const img of prescriptionImages) {
            if (!prescriptionImageMap[img.prescription_id]) prescriptionImageMap[img.prescription_id] = [];
            prescriptionImageMap[img.prescription_id].push(img);
        }

        // Step 5: Fetch shared reports (with prescription)
        const reports = await connection.query(
            `SELECT id, title, test_name, delivery_date, prescription_id, created_at
       FROM reports
       WHERE prescription_id = ANY($1::int[]) AND user_id = $2 AND shared = true AND deleted = false
       ORDER BY created_at DESC`,
            [prescriptionIds, user_id]
        );

        const reportIds = reports.map(r => r.id);

        const reportImages = await connection.query(
            `SELECT report_id, id as report_img_id, resiged, thumb
       FROM report_images
       WHERE report_id = ANY($1::int[]) AND deleted = false
       ORDER BY created_at ASC`,
            [reportIds]
        );

        const reportImageMap = {};
        for (const img of reportImages) {
            if (!reportImageMap[img.report_id]) reportImageMap[img.report_id] = [];
            reportImageMap[img.report_id].push(img);
        }

        const reportsByPrescription = {};
        for (const report of reports) {
            report.images = reportImageMap[report.id] || [];
            if (!reportsByPrescription[report.prescription_id]) {
                reportsByPrescription[report.prescription_id] = [];
            }
            reportsByPrescription[report.prescription_id].push(report);
        }

        // Step 6: Combine reports into prescriptions
        const combined = prescriptions.map(p => ({
            ...p,
            images: prescriptionImageMap[p.id] || [],
            reports: reportsByPrescription[p.id] || []
        }));

        // Step 7: Standalone shared reports (no prescription_id)
        const standaloneReports = await connection.query(
            `SELECT id, title, test_name, delivery_date, created_at
       FROM reports
       WHERE prescription_id IS NULL AND user_id = $1 AND shared = true AND deleted = false
       ORDER BY created_at DESC`,
            [user_id]
        );

        const standaloneReportIds = standaloneReports.map(r => r.id);

        const standaloneImages = await connection.query(
            `SELECT report_id, id as report_img_id, resiged, thumb
       FROM report_images
       WHERE report_id = ANY($1::int[]) AND deleted = false
       ORDER BY created_at ASC`,
            [standaloneReportIds]
        );

        const standaloneImageMap = {};
        for (const img of standaloneImages) {
            if (!standaloneImageMap[img.report_id]) standaloneImageMap[img.report_id] = [];
            standaloneImageMap[img.report_id].push(img);
        }

        for (const report of standaloneReports) {
            report.images = standaloneImageMap[report.id] || [];
        }

        // Step 8: Return everything
        return res.status(200).json({
            flag: 200,
            accessToken: fileAccessToken,
            prescriptions: combined,
            standaloneReports,
            message: 'Shared documents fetched successfully.'
        });

    } catch (error) {
        console.error('Error in getSharedDocs:', error);
        return res.status(500).json({ error: 'An unknown error occurred.' });
    } finally {
        await connection.release();
    }
});









app.post('/uploadProfile', upload.single('image'), async (req, res) => {

    const connection = await database.getConnection(); // Get DB connection
    try {
        const { accessToken, member_id } = req.body;
        // Check if files are uploaded
        if (!member_id) {
            return res.status(400).json({ error: 'Missing field!' });
        }
        if (!(Number.isInteger(+member_id) && +member_id > 0)) {
            return res.status(400).json({ error: 'invalid type!' });

        }
        // Decode token and verify user existence
        const { decodedToken, isExist } = await authintication(accessToken, member_id, connection);


        // Check if files are uploaded
        if (!req.file) {
            return res.status(400).json({ error: 'No files uploaded.' });
        }

        await connection.beginTransaction(); // Start a transaction
        const processedImages = await processImages([req.file], decodedToken.userId);
        // const invalidImages = [];
        const uploadFolder = path.join(__dirname, 'profiles');

        await fs.mkdir(uploadFolder, { recursive: true });



        const { color } = processedImages[0];


        // Save color image for frontend
        const colorPath = path.join(uploadFolder, color.filename);
        await fs.writeFile(colorPath, color.buffer);

        await connection.queryOne(
            `UPDATE users SET profile_image_url = $1 WHERE user_id = $2`,
            [`/profiles/${color.filename}`, member_id]
        );


        // Delete previous profile image if exists
        if (isExist.profile_image_url) {
            const previousPath = path.join(__dirname, isExist.profile_image_url);
            try {
                await fs.unlink(previousPath);
            } catch (err) {
                if (err.code !== 'ENOENT') {
                    console.error('Failed to delete previous image:', err);
                    throw new Error('Error deleting old profile image.');
                }
                // ENOENT means file doesn't exist, which is fine
            }
        }


        await connection.commit(); // Commit transaction
        res.status(200).json({ message: 'Profile pic uploaded successfully.' });
    } catch (error) {

        await connection.rollback();

        if (error instanceof multer.MulterError) {


            if (error.code === 'LIMIT_FILE_COUNT') {
                return res.status(400).json({ error: 'You can only upload a maximum of 10 files.' });
            } else if (error.code === 'LIMIT_FILE_SIZE') {
                return res.status(400).json({ error: 'File size exceeds the limit.' });
            }
        } else if (error.message === 'Access token is required.' || error.message === 'Invalid or expired access token.' || error.message === 'Invalid user.') {
            return res.status(403).json({ error: error.message });
        } else {
            return res.status(500).json({ error: 'An unknown error occurred.' });
        }

        console.error('Error processing images:', error);
    } finally {
        await connection.release(); // Release DB connection
    }
});








app.listen(port, '0.0.0.0', () => {
    console.log(`Server running on port ${port}`);
});