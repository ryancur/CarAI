# Generated by Django 4.0.3 on 2023-09-22 21:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inventory_rest', '0005_alter_suggestedvaluevo_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='suggestedvaluevo',
            name='suggested_value',
            field=models.DecimalField(decimal_places=2, max_digits=8),
        ),
    ]