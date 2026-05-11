use base64::{engine::general_purpose::URL_SAFE, Engine as _};
use image::ImageReader;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    io::{BufRead, Cursor, Seek},
};
#[derive(Clone, Debug)]
pub struct SonicCCClient {
    client: Client,
    host: Option<String>,
    api_key: Option<String>,
    frequency_penalty: f64,
    temperature: f64,
    top_p: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChatMessage {
    System(String),
    User(Content),
    Assistant(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Content {
    text: String,
    image: Option<String>,
    audio: Option<String>,
}

impl Content {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            image: None,
            audio: None,
        }
    }
    pub fn with_image_data(&self, img: impl BufRead + Seek) -> Self {
        Self {
            text: self.text.clone(),
            image: Some(image_data_to_image_text(img)),
            audio: None,
        }
    }
}

impl SonicCCClient {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            host: None,
            api_key: None,
            frequency_penalty: 0.0,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
    pub fn with_host(&mut self, host: impl Into<String>) -> Self {
        self.host = Some(format!("http://{}/v1/chat/completions", host.into()));
        self.clone()
    }
    pub fn with_api_key(&mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self.clone()
    }
    pub fn set_temperature(&mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self.clone()
    }
    pub fn set_top_p(&mut self, top_p: f64) -> Self {
        self.top_p = top_p;
        self.clone()
    }
    pub async fn get_chat_completion(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<String, reqwest::Error> {
        let json_messages: Vec<Value> = messages
            .into_iter()
            .map(|x| match x {
                ChatMessage::System(content) => {
                    let mut message_hashmap: HashMap<String, String> = HashMap::new();
                    message_hashmap.insert("role".to_string(), "system".to_string());
                    message_hashmap.insert("content".to_string(), content.clone());
                    serde_json::to_value(&message_hashmap).unwrap()
                }
                ChatMessage::Assistant(content) => {
                    let mut message_hashmap: HashMap<String, String> = HashMap::new();
                    message_hashmap.insert("role".to_string(), "assistant".to_string());
                    message_hashmap.insert("content".to_string(), content.clone());
                    serde_json::to_value(&message_hashmap).unwrap()
                }
                ChatMessage::User(content) => {
                    let mut content_vec: Vec<Value> = Vec::new();
                    let mut text: HashMap<String, String> = HashMap::new();
                    text.insert("type".to_string(), "text".to_string());
                    text.insert("text".to_string(), content.text.clone());
                    content_vec.push(serde_json::to_value(text).unwrap());
                    if let Some(image_text) = content.image {
                        let mut img_content: HashMap<String, String> = HashMap::new();
                        img_content.insert("type".to_string(), "input_image".to_string());
                        img_content.insert("image_url".to_string(), image_text);
                        content_vec.push(serde_json::to_value(img_content).unwrap());
                    }
                    let mut message_hashmap: HashMap<String, Value> = HashMap::new();
                    message_hashmap.insert(
                        "role".to_string(),
                        serde_json::to_value("user".to_string()).unwrap(),
                    );
                    message_hashmap.insert(
                        "content".to_string(),
                        serde_json::to_value(content_vec).unwrap(),
                    );
                    serde_json::to_value(message_hashmap).unwrap()
                }
            })
            .collect();
        /*
        let out_json = json!({
            "model": "ggml_llava-v1.5-13b",
            "messages": json_messages,
            "temperature": 0.7,
            "frequency_penalty": self.frequency_penalty
        });
        */
        let out_json = json!({
            "model": "default",
            "messages": json_messages,
            "temperature": self.temperature,
            "enable_thinking": false,
            "top_p": self.top_p
            //"frequency_penalty": self.frequency_penalty
        });
        let response_value: serde_json::Value = self
            .client
            .post(self.host.as_ref().unwrap())
            .json(&out_json)
            .send()
            .await?
            .json()
            .await?;
        Ok(response_value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string())
    }
}

pub fn image_data_to_image_text(img: impl BufRead + Seek) -> String {
    let reader = ImageReader::new(img).with_guessed_format().unwrap();
    let input_image = reader.decode().unwrap();
    let mut png_data: Vec<u8> = Vec::new();
    input_image
        .write_to(&mut Cursor::new(&mut png_data), image::ImageFormat::Png)
        .unwrap();
    format!(
        "data:image/png;base64,{}",
        URL_SAFE.encode(png_data.as_slice())
    )
}
