import React from "react";
import { TouchableOpacity, Text, StyleSheet } from "react-native";

type ButtonProps = {
	title: string;
	onPress: () => void;
	disabled?: boolean;
};

export function CustomButton({ title, onPress, disabled }: ButtonProps) {
	return (
		<TouchableOpacity 
			style={[styles.button, disabled && styles.buttonDisabled]} 
			onPress={onPress} 
			disabled={disabled}
		>
			<Text style={[styles.text, disabled && styles.textDisabled]}>{title}</Text>
		</TouchableOpacity>
	);
}

const styles = StyleSheet.create({
	button: {
		backgroundColor: '#007bff',
		padding: 12,
		borderRadius: 6,
		alignItems: 'center',
		marginVertical: 8,
	},
	buttonDisabled: {
		backgroundColor: '#cccccc',
		opacity: 0.6,
	},
	text: {
		color: '#fff',
		fontWeight: 'bold',
		fontSize: 16,
	},
	textDisabled: {
		color: '#999',
	},
});
