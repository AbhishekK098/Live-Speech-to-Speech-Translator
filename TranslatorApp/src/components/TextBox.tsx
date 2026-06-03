import React from "react";
import { View, Text, TextInput, StyleSheet } from "react-native";

type TextBoxProps = {
	label: string;
	value: string;
};

export function TextBox({ label, value }: TextBoxProps) {
	return (
		<View style={styles.container}>
			<Text style={styles.label}>{label}</Text>
			<TextInput
				style={styles.input}
				value={value}
				editable={false}
				multiline
			/>
		</View>
	);
}

const styles = StyleSheet.create({
	container: { marginBottom: 12 },
	label: { fontSize: 14, marginBottom: 4 },
	input: {
		borderWidth: 1,
		borderColor: '#ccc',
		borderRadius: 4,
		padding: 8,
		fontSize: 16,
		backgroundColor: '#f9f9f9',
	},
});
