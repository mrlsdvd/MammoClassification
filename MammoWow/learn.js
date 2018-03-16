'use strict';

var currNumber = 0;
var mammogramPictures  = ["2_ben_0.png", "3_ben_0.png", "3_ben_1.png", "3_ben_2.png", "4_ben_0.png", "4_ben_1.png", "4_ben_2.png", "4_mal_0.png", "4_mal_1.png", "4_mal_2.png", "4_mal_3.png", "5_mal_0.png", "5_mal_1.png"];
var mammogramSolutions = [2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5]
var mammogramMalignancy = ["Benign", "Benign", "Benign", "Benign", "Benign", "Benign", "Benign", "Malignant", "Malignant","Malignant","Malignant","Malignant","Malignant"]

function verifyPhoto(){

	var x = document.getElementById("userGuessSelection").value;

	if(x == mammogramSolutions[currNumber]){

		document.getElementById("answer").innerHTML = "Correct. " + "(" + mammogramMalignancy[currNumber] + ")";	
	}

	else{

		window.alert("Incorrect!");
		document.getElementById("answer").innerHTML = "Incorrect";	

	}
	
}

function nextPhoto(){
	document.getElementById("answer").innerHTML = "";	
	var randomNum = Math.floor(Math.random() * mammogramPictures.length);
	while(currNumber == randomNum){

		randomNum = Math.floor(Math.random() * mammogramPictures.length);

	}
	
	currNumber = randomNum;
	document.getElementById("picture").src = mammogramPictures[randomNum];

}

function firstLoad(){

	var randomNum = Math.floor(Math.random() * mammogramPictures.length);
	currNumber = randomNum;
	document.getElementById("picture").src = mammogramPictures[randomNum];


}

