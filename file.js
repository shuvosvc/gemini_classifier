const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const fs = require('fs').promises;
const path = require('path');
const Tesseract = require('tesseract.js');
const app = express();

require('dotenv').config();
const port = process.env.PORT || 7000;

const storage = multer.memoryStorage();

const projectUploadPath = path.join(__dirname, 'uploads'); 


app.use('/uploads', express.static(projectUploadPath));


const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    // Allow only JPEG, PNG, or WebP files
    const allowedTypes = /jpeg|jpg|png|webp/;
    const fileExt = file.originalname.split('.').pop().toLowerCase();
    const isValidType = allowedTypes.test(fileExt);
    if (isValidType) {
      return cb(null, true);
    } else {
      console.log("test");
      
      return cb(new Error('Invalid file type. Only JPEG, PNG, or WebP allowed.'));
    }
  }
});



const generateResizedImages = async (fileBuffer, originalFilename) => {
  const currentDate = new Date().toISOString().replace(/[-:]/g, ''); // Format: YYYYMMDDTHHMMSS
  const original = sharp(fileBuffer);


  const processedImg = original
    .grayscale()
    .resize(1024, 1024, { fit: 'inside' }) 
    .sharpen({ sigma: 1.2 }) 
    .modulate({ contrast: 2 }) 
    .toFormat('webp', { quality: 100 }); 

  const processedImgBuffer = await processedImg.toBuffer();

  const resizedImages = {
    classifyImg: { buffer: processedImgBuffer, filename: `${originalFilename}-classifyImg-${currentDate}.webp` },
  };

  return resizedImages;
};


const uploadToProjectFolder = async (resizedImages) => {
  const projectUploadPath = path.join(__dirname, 'uploads');
  await Promise.all(
    Object.values(resizedImages).map(async (image) => {
      const filePath = path.join(projectUploadPath, image.filename);
      await fs.writeFile(filePath, image.buffer);
    })
  );
};


const medicalKeywords = [
  
  "prescription", "doctor", "physician", "surgeon", "nurse", "hospital", "clinic", "patient", "medical", "diagnosis", "treatment", 
  "therapy", "medication", "meds", "medications", "clinical", "hospitalization", "health", "healthcare", "medical report", "treatment plan",
  "medical history", "outpatient", "inpatient", "referral", "consultation", "consulting", "check-up", "vaccination", "screening",
  "routine", "discharge", "medical certificate", "lab test", "test result", "test report", "health condition", "medical condition",

  // Doctors and Medical Professionals
  "dr", "dr.", "doctor", "physician", "surgeon", "cardiologist", "neurologist", "orthopedist", "dermatologist", "pediatrician", 
  "gynecologist", "radiologist", "pathologist", "oncologist", "urologist", "dentist", "psychiatrist", "optometrist", "therapist",
  "chiropractor", "podiatrist", "dentist", "ENT", "audiologist", "speech therapist", "clinical psychologist", "nurse practitioner",

  // Diagnostic and Test-related Terms
  "blood test", "cbc", "lipid profile", "glucose test", "blood sugar", "esr", "thyroid function test", "liver function test", "kidney function test", 
  "urine test", "stool test", "sputum test", "biopsy", "ct scan", "mri", "ultrasound", "ecg", "x-ray", "mammogram", "pet scan", "pcr", 
  "ultrasound scan", "endoscopy", "colonoscopy", "serum test", "hba1c", "prostate test", "glucose monitoring", "cholesterol level", 
  "clinical examination", "blood pressure", "body temperature", "pulse rate", "oxygen saturation", "imaging", "microbiology", "cytology", 
  "histopathology", "electrocardiogram", "electromyography", "biochemical tests",

  // Medical Conditions and Diseases
  "diabetes", "hypertension", "heart disease", "stroke", "cancer", "breast cancer", "prostate cancer", "liver cancer", "lung cancer", "colon cancer", 
  "tuberculosis", "asthma", "arthritis", "migraine", "chronic pain", "back pain", "osteoporosis", "rheumatoid arthritis", "autoimmune disease", 
  "alzheimers", "parkinsons", "dementia", "chronic obstructive pulmonary disease", "kidney failure", "renal disease", "hepatitis", "HIV", "AIDS", 
  "covid-19", "covid", "flu", "influenza", "cold", "virus", "infection", "pneumonia", "sepsis", "malaria", "tuberculosis", "HIV", "syphilis",
  "hepatitis B", "hepatitis C", "cholera", "dengue", "diabetic retinopathy", "cirrhosis", "hepatitis A", "gout", "bipolar disorder", "depression", 
  "anxiety", "mental health", "psychiatric disorders", "schizophrenia", "bipolar", "depressive disorder", "psychosis", "post-traumatic stress disorder",

  // Medications and Treatments
  "painkiller", "analgesic", "antibiotic", "antiviral", "antifungal", "pain relief", "insulin", "glucagon", "antidepressant", "antihypertensive", 
  "beta blocker", "antibiotics", "steroids", "chemo", "chemotherapy", "radiotherapy", "vaccination", "immunization", "antihistamine", 
  "medication list", "pill", "tablet", "capsule", "syrup", "ointment", "cream", "inhaler", "nasal spray", "intravenous", "vaccine", 
  "immunotherapy", "dialysis", "blood transfusion", "surgical procedure", "organ transplant", "anesthesia", "surgical removal", "anticoagulants",
  "aspirin", "penicillin", "metformin", "ibuprofen", "paracetamol", "morphine", "antipsychotic", "anticonvulsant", "antibiotics", "statin", "insulin",
  
  // Common Medical Procedures
  "surgery", "operation", "cataract surgery", "appendectomy", "heart surgery", "spinal surgery", "knee replacement", "hip replacement", 
  "bypass surgery", "cesarean section", "plastic surgery", "cosmetic surgery", "laparoscopy", "stitch", "surgical incision", "bone marrow biopsy", 
  "organ transplant", "dialysis", "endoscopic procedure", "bronchoscopy", "colonoscopy", "arthroscopy", "mammoplasty", "bariatric surgery",
  
  // Health-related Terms
  "obesity", "weight loss", "weight management", "nutrition", "diet", "exercise", "rehabilitation", "chronic condition", "wellness", "physical therapy", 
  "rehab", "fitness", "strengthening", "mobility", "cardiopulmonary", "diabetes management", "hypertension management", "post-surgery care", 
  "pain management", "blood pressure control", "mental health care", "nursing care", "home care", "dietician", "nutritionist", "pediatric care", 
  "geriatric care", "senior care", "geriatric care",

  // Other Related Terms
  "medical imaging", "laboratory", "pharmacy", "medicines", "healthcare provider", "emergency room", "urgent care", "clinical trial", 
  "drug interaction", "medical advice", "symptom", "diagnostic test", "prescription refill", "patient care", "care plan", "medical insurance",
  "health insurance", "health check", "preventive care", "clinical notes", "follow-up", "medical documentation", "health records", 
  "treatment record", "patient history", "sick leave", "medical leave", "physical examination", "medical report"
];



const extractTextAndClassify = async (filePath) => {
  try {
    const { data: { text } } = await Tesseract.recognize(filePath, 'eng', { logger: (m) => console.log(m) });


    const cleanedText = text.toLowerCase().replace(/\s+/g, ' ').trim();

    
    const foundKeywords = medicalKeywords.filter(keyword => cleanedText.includes(keyword));

  
    const isMedicalDocument = foundKeywords.length > 0;

    return { extractedText: text, isMedicalDocument };
  } catch (error) {
    console.error("Error during OCR processing:", error);
    throw new Error('Error processing the document.');
  }
};


app.post('/api/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).send('No file uploaded.');
    }

    const originalFilename = req.file.originalname.split('.')[0];
    const resizedImages = await generateResizedImages(req.file.buffer, originalFilename);

    
    await uploadToProjectFolder(resizedImages);

   
    const baseUrl = process.env.baseUrl || `http://localhost:${port}/uploads`;
    const imageUrls = {
      classifyImg: `${baseUrl}/${resizedImages.classifyImg.filename}`,
    };

   
    const result = await extractTextAndClassify(`${projectUploadPath}/${resizedImages.classifyImg.filename}`);


    res.json({
      imageUrl: imageUrls.classifyImg,
      extractedText: result.extractedText,
      isMedicalDocument: result.isMedicalDocument ? 'Yes' : 'No',
    });
  } catch (error) {
    console.error('Error processing/uploading image:', error);
    res.status(500).send('Internal Server Error');
  }
});
























const deleteImagesByPaths = async (imagePaths) => {
    const baseUrlRegex = /^(https?:\/\/[^/]+\/uploads\/)/; // Match the base URL
  
    await Promise.all(
      imagePaths.map(async (imagePath) => {
        try {
          // Remove the base URL and the "uploads/" prefix from the path using regex
          const imagePathWithoutBaseUrl = imagePath.replace(baseUrlRegex, '');
  
          // Ensure the path is within the designated projectUploadPath to prevent unintended deletions
  
          const absolutePath = path.resolve(projectUploadPath, imagePathWithoutBaseUrl);
  
          if (absolutePath.startsWith(projectUploadPath)) {
            // Delete the image
            await fs.unlink(absolutePath);
          } else {
            console.warn(`Invalid path: ${imagePathWithoutBaseUrl}`);
          }
        } catch (error) {
          console.error(`Error deleting image at path: ${imagePath}`, error);
        }
      })
    );
  };



  app.delete('/api/delete',  express.json(), async (req, res) => {
    try {
      const { classifyImg } = req.body;
  
      if ( !classifyImg) {
        return res.status(400).send('Invalid request format. Ensure all four image paths are provided.');
      }
  
      // Delete images based on the provided paths
      await deleteImagesByPaths([ classifyImg]);
  
      res.status(200).send('Images deleted successfully.');
    } catch (error) {
  
      res.status(500).send('Internal Server Error');
    }
  });

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});